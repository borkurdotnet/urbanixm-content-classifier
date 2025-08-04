import argparse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import gzip
import json
import pickle
import jsonlines
import logging
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import shuffle
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

POSITIVE_COUNT_MIN = 20
POSITIVE_TEST_COUNT_MIN = 10
MINIMUM_TRAINING_INSTANCES = 25


class ObjectiveType(Enum):
    ON_TOPIC = "on_topic"
    QUOTABLE = "quotable"
    TOPICS = "topics"
    PLACES = "places"

@dataclass
class Texts:
    positive_direct: list[str] = field(default_factory=list)
    positive_indirect: list[str] = field(default_factory=list)
    negative_direct: list[str] = field(default_factory=list)
    negative_indirect: list[str] = field(default_factory=list)

@dataclass
class DatasetCounts:
    positive_direct: int
    positive_indirect: int
    negative_direct: int
    negative_indirect: int
    positive_train: int
    positive_test: int
    negative_train: int
    negative_test:int

    def to_dict(self) -> dict[str, Any]:
        # Return a json serializable dict
        return {
            "positive_direct": self.positive_direct,
            "positive_indirect": self.positive_indirect,
            "negative_direct": self.negative_direct,
            "negative_indirect": self.negative_indirect,
            "positive_train": self.positive_train,
            "positive_test": self.positive_test,
            "negative_train": self.negative_train,
            "negative_test": self.negative_test
        }

@dataclass
class Dataset:
    counts: DatasetCounts
    data_train: pd.DataFrame
    data_test: pd.DataFrame
    #FIXME: Add validation

@dataclass
class EvaluationResult:
    accuracy: float
    count: int

@dataclass
class Evaluation:
    train: EvaluationResult
    test: EvaluationResult

    def to_dict(self) -> dict[str, Any]:
        # Return a json serializable dict
        return {
            "train": {
                "accuracy": self.train.accuracy,
                "count": self.train.count
            },
            "test": {
                "accuracy": self.test.accuracy,
                "count": self.test.count
            }
        }

@dataclass
class LabelledEvaluation:
    label: str
    evaluation: Evaluation

    def to_dict(self) -> dict[str, Any]:
        # Return a json serializable dict
        return {
            "label": self.label,
            "evaluation": self.evaluation.to_dict()
        }

@dataclass
class EvaluatedModel:
    model: Any
    counts: DatasetCounts
    evaluations: list[LabelledEvaluation]

    def to_dict(self) -> dict[str, Any]:
        # Return a json serializable dict
        return {
            "counts": self.counts,
            "evaluations": [e.to_dict() for e in self.evaluations]
        }

class ArticleClassificationTrainer(object):
    data_dir: str
    objective_type: ObjectiveType
    training_data_path: str
    training_metadata_path: str
    output_model_path: str

    def __init__(self):
        pass

    def parse_arguments_training(self) -> None:
        """
        Parses command line arguments and set the appropiate class variables
        """
        parser = argparse.ArgumentParser(
            description="Trainer for quote multilabel classification")
        parser.add_argument("--data_dir", 
                            type=str, 
                            required=True, 
                            help="Path to the directory containing training data and output model")
        parser.add_argument("--objective_type", 
                            type=str, 
                            required=True, 
                            help="Objective: [on_topic, quotable, topic, place]")
        args = parser.parse_args()

        # Initialize data directory
        self.data_dir = args.data_dir
        if not os.path.isdir(self.data_dir):
            # Argument is not a directory
            logging.error(f"--data_dir argument is not a directory: {self.data_dir}")
            exit(1)

        # Initialize objective type
        valid_objective_types = [obj_type.value for obj_type in ObjectiveType]
        if args.objective_type not in valid_objective_types:
            logging.error(f"Unknown value for --objective_type: {args.objective_type}")    
            exit(1)
        self.objective_type = ObjectiveType(args.objective_type)

        # Set training data path
        self.training_data_path = os.path.join(self.data_dir, 
                                               "classifiers", 
                                               "training-data", 
                                               "articles.jsonl.gz")
        self.training_metadata_path = os.path.join(self.data_dir, 
                                                   "classifiers", 
                                                   "training-data", 
                                                   "articles_meta.json")

        # Set output model path
        self.output_model_path = os.path.join(self.data_dir, 
                                              "classifiers", 
                                              "models", 
                                              "final", 
                                              f"article_models")

    def load_data(self, objective_label: str | None) -> Texts:
        """
        Load the training and testing data

        :param objective_label: The topic or place label, when appropriate
        """
        
        # Load the training data
        if os.path.exists(self.training_data_path):
            fp = gzip.open(self.training_data_path)
        else:
            self.training_data_path = self.training_data_path.removesuffix(".gz")
            fp = open(self.training_data_path)

        json_reader = jsonlines.Reader(fp)

        texts = Texts()
        if self.objective_type == ObjectiveType.ON_TOPIC or self.objective_type == ObjectiveType.QUOTABLE:
            # We are traing a model for estimating if a web-page is on the general topic of urbanism
            # or a model for estimating if an urbanism web-page is content rich (quotable)
            texts =  self.load_data_unlabelled(json_reader)
        elif self.objective_type == ObjectiveType.TOPICS and objective_label is not None:
            # We are training a model for estimating if a web-page is on a sepecific urbanism topic
            texts = self.load_data_labelled(objective_label, json_reader)
        elif self.objective_type == ObjectiveType.PLACES and objective_label is not None:
            # We are training a model for estimating if a web-page is talking about a specific plce
            texts = self.load_data_labelled(objective_label, json_reader)
        else:
            print(f"Unkonwn objective type or label: {self.objective_type}/{objective_label}")

        json_reader.close()
        fp.close()
        return texts

    def load_data_unlabelled(self, json_reader: jsonlines.Reader) -> Texts:
        """
        Load data for the two special classification classes 'on-topic' and 'quotable',
        which consider, respecively, any article on urbanism to be positive 
        or any urbanism article that is text-rich

        :param json_reader:
        :return:
        """
        texts = Texts()
        for json_object in json_reader:
            text = json_object['content']
            if json_object['off_topic']:
                # Article is off topic
                texts.negative_direct.append(text)
            else:
                # Article is on topic
                if self.objective_type == ObjectiveType.QUOTABLE and json_object['un_quotable']:
                    # Article is on topic but unquotable (archive or something like that)
                    texts.negative_indirect.append(text)
                else:
                # Article is on topic and quotable
                    texts.positive_direct.append(text)
        return texts

    def load_data_labelled(self, objective_label: str, json_reader: jsonlines.Reader) -> Texts:
        """
        Loads the data from the json_reader creates appropriate training and testing texts
        given the topic/place label being processed

        :param objective_label: The label of the topic or the place being processed
        :param json_reader: A reader for the jsonlines file with input data

        :returns: A Texts object with the texts from the input data
        """
        texts = Texts()
        for json_object in json_reader:
            text = json_object['content']
            if json_object['off_topic']:
                # Article is not about urbanism
                texts.negative_direct.append(text)
            else:
                # Article is on topic
                if objective_label in json_object[self.objective_type.value]["direct"]:
                    # Article has been annotated directly on topic
                    texts.positive_direct.append(text)
                elif objective_label in json_object[self.objective_type.value]["derived"]:
                    # Article has been derived to be on topic
                    texts.positive_indirect.append(text)
                else:
                    # Article is about urbanism but not on the desired objective label (topic or place)
                    texts.negative_indirect.append(text)
        return texts

    def get_data_spit(self, texts: Texts) -> Dataset:
        """
        Split the task's training data in to train/valdation/test sets

        :param texts: is a collection of texts used for training and testing

        :return: Dataset where data fields are dataframes with columns ['text','label','weight','p/n','d/i']

        where:
            - __text__ is the text of an article
            - __label__ is 0 if negative and 1 if positive
            - __weight__ is the weight based on the negative/positive and direct/indirect values
            - __p/n__ 'positive' or 'negative'
            - __d/i__ 'direct' or 'indirect'
        """

        # Choose test count as 10% of positive cases
        pos_test_count = max(POSITIVE_TEST_COUNT_MIN,
                             int(.1 * (len(texts.positive_direct)+len(texts.positive_indirect))))
        pos_neg_ratio = (len(texts.positive_direct) + len(texts.positive_indirect)) \
            / (len(texts.negative_direct) + len(texts.negative_indirect))

        # Positive direct samples
        pos_train_df = pd.DataFrame(data=texts.positive_direct, columns=['text'])
        if len(texts.positive_direct) > 0:
            pos_train_df['label'] = 1
            pos_train_df['weight'] = 1.0
            pos_train_df['p/n'] = 'positive'
            pos_train_df['d/i'] = 'direct'

        # Positive indirect samples (with 75% weight)
        posi_train_df = pd.DataFrame(data=texts.positive_indirect, columns=['text'])
        if len(texts.positive_indirect) > 0:
            posi_train_df['label'] = 1
            posi_train_df['weight'] = 0.75
            posi_train_df['p/n'] = 'positive'
            posi_train_df['d/i'] = 'indirect'

        # Positive train/test split
        pos_train_df = pd.concat([pos_train_df, posi_train_df], ignore_index=True)
        pos_test_df = pos_train_df.sample(n=pos_test_count)
        pos_train_df = pos_train_df.drop(pos_test_df.index)

        # Negative direct samples
        neg_train_df = pd.DataFrame(data=texts.negative_direct, columns=['text'])
        if len(texts.negative_direct) > 0:
            neg_train_df['label'] = 0
            neg_train_df['weight'] = pos_neg_ratio
            neg_train_df['p/n'] = 'negative'
            neg_train_df['d/i'] = 'direct'

        # Negative indirect samples (with weight equal to the ratio betwee positive and negative data)
        negi_train_df = pd.DataFrame(data=texts.negative_indirect, columns=['text'])
        if len(texts.negative_indirect) > 0:
            negi_train_df['label'] = 0
            negi_train_df['weight'] = pos_neg_ratio
            negi_train_df['p/n'] = 'negative'
            negi_train_df['d/i'] = 'indirect'

        # Negative train/test split
        neg_train_df = pd.concat([neg_train_df, negi_train_df], ignore_index=True)
        if neg_train_df.shape[0] > pos_test_count:
            neg_test_df = neg_train_df.sample(n=pos_test_count)
        else:
            neg_test_df = neg_train_df.copy()
        neg_train_df = neg_train_df.drop(neg_test_df.index)

        dataset = Dataset(
            counts=DatasetCounts(
                positive_direct=len(texts.positive_direct),
                positive_indirect=len(texts.positive_indirect),
                negative_direct=len(texts.negative_direct),
                negative_indirect=len(texts.negative_indirect),
                positive_train=pos_train_df.shape[0],
                positive_test=pos_test_count,
                negative_train=neg_train_df.shape[0],
                negative_test=neg_test_df.shape[0]
            ),
            data_train=pd.DataFrame(shuffle(pd.concat([pos_train_df, neg_train_df]))),
            data_test=pd.DataFrame(shuffle(pd.concat([pos_test_df, neg_test_df])))
        )

        return dataset

    def train_model(self, texts: Texts) -> EvaluatedModel:
        """
        Takes a set of text objects and trains a model.
        The model is evaluated along several metrics to get a complete understanding of the model performance.
        """

        training_data = self.get_data_spit(texts)

        parameters = {
            'vect__ngram_range': [(1, 1)],
            'tfidf__use_idf': [True],
        }

        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SVC(probability=True))
                             ])
        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

        gs_clf = gs_clf.fit(list(training_data.data_train['text']),
                            training_data.data_train['label'],
                            clf__sample_weight=training_data.data_train['weight'])

        collected_evaluation: list[LabelledEvaluation] = []

        # Overall accuracy
        overall_evaluation = LabelledEvaluation(
            label='overall',
            evaluation = Evaluation(
                train=EvaluationResult(
                    accuracy=float(gs_clf.score(list(training_data.data_train['text']),
                                                training_data.data_train['label'])),
                    count=training_data.data_train.shape[0]
                ),
                test=EvaluationResult(
                    accuracy=float(gs_clf.score(list(training_data.data_test['text']),
                                                training_data.data_test['label'])),
                    count=training_data.data_test.shape[0]
                )
            )
        )
        collected_evaluation.append(overall_evaluation)

        # Source accuracy
        for sign in ['positive', 'negative']:
            for direction in ['direct', 'indirect']:

                train_sub = training_data.data_train[training_data.data_train['p/n'] == sign]
                train_sub = train_sub[train_sub['d/i'] == direction]
                test_sub = training_data.data_test[training_data.data_test['p/n'] == sign]
                test_sub = test_sub[test_sub['d/i'] == direction]

                if train_sub.shape[0] > 0:
                        evaluation = LabelledEvaluation(
                            label=f"{sign}_{direction}",
                            evaluation=Evaluation(
                                train=EvaluationResult(
                                    accuracy=float(gs_clf.score(list(train_sub['text']), 
                                                          train_sub['label']))
                                                          if train_sub.shape[0] > 0 else 0.0,
                                    count=train_sub.shape[0]
                                ),
                                test=EvaluationResult(
                                    accuracy=float(gs_clf.score(list(test_sub['text']),
                                                                test_sub['label'])) 
                                                                if test_sub.shape[0] > 0 else 0.0,
                                    count=test_sub.shape[0]
                                )
                            )
                        )
                        collected_evaluation.append(evaluation)

        return EvaluatedModel(
            model=gs_clf,
            counts=training_data.counts,
            evaluations=collected_evaluation,
        )

    def save_classifier_model(self, 
                              objective_label: str | None, 
                              evaluated_model: EvaluatedModel) -> None:
        """
        Save classifier model and its evaluation meta-data
        :param objective_label: For topics and places, the topic or place label
        :param evaluated_model: An object containing both the model and the meta-data
        """

        # Save classifier meta-data
        metadata =  {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'objective_type': self.objective_type.value,
            'objective_label': objective_label,
            'counts': evaluated_model.counts.to_dict(),
            'evaluations': [e.to_dict() for e in evaluated_model.evaluations]
        }
        model_meta_filename = self.objective_type.value
        if objective_label is not None:
            model_meta_filename += f"_{objective_label}"
        model_meta_filename += ".json"
        model_meta_path = os.path.join(self.output_model_path, self.objective_type.value)
        os.makedirs(model_meta_path, exist_ok=True)
        model_meta_path = os.path.join(model_meta_path, model_meta_filename)

        with open(model_meta_path, 'w') as fout:
            fout.write(json.dumps(metadata, indent=2))
            fout.close()

        # Save classifier model pickle
        model_pickle_path = model_meta_path.replace('.json', '.pickle')
        with open(model_pickle_path, 'wb') as fh:
            pickle.dump(evaluated_model.model, fh)
            fh.close()

    def train_models(self) -> None:
        """
        Trains the appropriate models using information passed on the command line
        """

        # Global models
        if (self.objective_type == ObjectiveType.ON_TOPIC 
            or self.objective_type == ObjectiveType.QUOTABLE):

            # Train a single model for either the on-topic class or quotable class
            texts = self.load_data(objective_label=None)
            evaluated_model = self.train_model(texts)
            self.save_classifier_model(objective_label=None, 
                                       evaluated_model=evaluated_model)
            
            return
        
        # Read training data meta-data to get information about training data counts
        # for individual topics and places
        with open(self.training_metadata_path) as fh:
            training_data_metadata = json.load(fh)
            fh.close()

        if self.objective_type == ObjectiveType.TOPICS:
            # Train a classifier for each topic with sufficient training data
            for topic_label, topic_stats in training_data_metadata['topics'].items():
                # Train a classifier for topic with label topic_label
                training_instances = topic_stats['direct'] + topic_stats['derived']
                logging.info(f"Building model for {topic_label} ({training_instances} instances)")
                if training_instances < MINIMUM_TRAINING_INSTANCES:
                    logging.warning("Topic has insufficient training data: {}".format(topic_label))
                    continue

                texts = self.load_data(objective_label=topic_label)
                evaluated_model = self.train_model(texts)
                self.save_classifier_model(objective_label=topic_label, 
                                        evaluated_model=evaluated_model)
            
        if self.objective_type == ObjectiveType.PLACES:
            # Train a classifier for each place with sufficient training data
            for place_label, place_stats in training_data_metadata['places'].items():
                # Train a classifier for place with label place_label
                training_instances = place_stats['direct'] + place_stats['derived']
                logging.info(f"Building model for {place_label} ({training_instances} instances)")
                if training_instances < MINIMUM_TRAINING_INSTANCES:
                    logging.warning("Place has insufficient training data: {}".format(place_label))
                    continue

                texts = self.load_data(objective_label=place_label)
                evaluated_model = self.train_model(texts)
                self.save_classifier_model(objective_label=place_label, 
                                        evaluated_model=evaluated_model)
        

if __name__ == "__main__":
    """
    Parses the arguments passed on the command line and trains the appropriate models
    """
    trainer = ArticleClassificationTrainer()
    trainer.parse_arguments_training()
    trainer.train_models()
