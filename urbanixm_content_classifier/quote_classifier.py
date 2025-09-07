import argparse
import csv
from datasets import Dataset, DatasetDict
from datetime import datetime
from enum import Enum
import functools
import gzip
import logging
import numpy as np
import json
import os
import random
from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split
import torch
import torch.nn.functional as F
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ObjectiveType(Enum):
    """Enumeration for the different types of quote classification objectives"""
    QUOTE_TYPES = 'quote_types'
    QUOTE_TONES = 'quote_tones'
    QUOTE_TOPICS = 'quote_topics'
    QUOTE_PLACES = 'quote_places'


class CustomTrainer(Trainer):
    """
    Custom trainer class to be able to pass label weights and calculate mutilabel loss
    """

    def __init__(self, label_weights, device, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights
        self.device = device
            
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").to(self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
                
        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32), pos_weight=self.label_weights)
        return (loss, outputs) if return_outputs else loss


class QuoteClassificationTrainer(object):
    base_model_name: str
    data_dir: str
    objective_type: ObjectiveType
    label_names: list[str]
    training_data_path: str
    interim_model_path: str
    output_model_path: str
    output_meta_path: str


    def __init__(self) -> None:
        self.base_model_name = 'distilroberta-base'  # Default base model to be used
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


    def parse_arguments_training(self) -> None:
        """
        Parses command line arguments
        """
        parser = argparse.ArgumentParser(
            description="Trainer for quote multilabel classification")
        parser.add_argument("--base_model",
                            type=str,
                            required=False,
                            help="Base model to use for classifier")
        parser.add_argument("--data_dir", 
                            type=str, 
                            required=True, 
                            help="Path to the directory containing training data and output model")
        parser.add_argument("--objective_type", 
                            type=str, 
                            required=True, 
                            help="Objective: [quote_types, quote_tones, quote_topics, quote_places]")
        args = parser.parse_args()

        # Initalize base model
        if args.base_model:
            self.base_model_name = args.base_model

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
                                        "classifiers", "training-data", 
                                        f"quotes_{self.objective_type.value}.csv")

        # Set output model path
        self.output_model_path = os.path.join(self.data_dir, 
                                            "classifiers", "models", "final", 
                                            f"quotes_{self.objective_type.value}_multilabel")
        self.output_meta_path = os.path.join(self.data_dir,
                                             "classifiers", "models", "final",
                                             f"quotes_{self.objective_type.value}_multilabel.json")
        
        # Set director for interim models used in training
        self.interim_model_path = os.path.join(self.data_dir,
                                                "classifiers", "models", "interim",
                                               f"quotes_{self.objective_type.value}_multilabel")


    @staticmethod
    def load_text_and_labels(file_path: str) -> tuple[list[str], tuple[Any], np.ndarray]:
        """
        Processes a training/test/evaluation data file and returns:
        - label_names: Ordered list of strings representing the names of the classification labels
        - text: Tuple of text samples
        - labels: NumPy array of binary labels
        """
        # set random seed for reproducibility
        random.seed(0)

        # load data
        if os.path.exists(f"{file_path}.gz"):
            logging.info(f"Loading compressed data from {file_path}.gz")
            with gzip.open(f"{file_path}.gz", 'rt') as csvfile:
                data = list(csv.reader(csvfile, delimiter=','))
        elif os.path.exists(file_path):
            logging.info(f"Loading data from {file_path}")
            with open(file_path) as csvfile:
                data = list(csv.reader(csvfile, delimiter=','))
        else:
            logging.error(f"No data file found: {file_path}[.gz]")
            exit(1)
        header_row = data.pop(0)
        label_names = header_row[1:]

        # shuffle data
        random.shuffle(data)

        # reshape
        text, labels = list(zip(*[(row[0], row[1:]) for row in data]))
        labels = np.array(labels, dtype=float)
        labels = labels.astype(int)

        return label_names, text, labels


    def load_data(self) -> tuple[np.ndarray, torch.Tensor, DatasetDict]:
        """
        Loads training data from a csv file of the format
        text, label-1, label-2, ...
        """

        label_names, text, labels = self.load_text_and_labels(self.training_data_path)
        self.label_names = label_names

        # create label weights, convert to float32 and move to device
        label_weights = 1 - labels.sum(axis=0) / labels.sum()
        label_weights = torch.tensor(label_weights, dtype=torch.float32, device=self.device)

        # stratified train test split for multilabel ds
        row_ids = np.arange(len(labels))
        train_idx, y_train, val_idx, y_val = iterative_train_test_split(row_ids[:,np.newaxis], labels, test_size = 0.1)
        x_train = [text[i] for i in train_idx.flatten()]
        x_val = [text[i] for i in val_idx.flatten()]

        # create hf dataset
        dataset_dict = DatasetDict({
            'train': Dataset.from_dict({'text': x_train, 'labels': y_train}),
            'eval': Dataset.from_dict({'text': x_val, 'labels': y_val})
        })

        return labels, label_weights, dataset_dict

    @staticmethod
    def tokenize_examples(examples: DatasetDict, tokenizer):
            tokenized_inputs = tokenizer(examples['text'])
            tokenized_inputs['labels'] = examples['labels']
            return tokenized_inputs

    @staticmethod
    def summarize_training(trainer: Trainer, 
                           dataset: DatasetDict, 
                           tokenized_dataset: DatasetDict,
                           label_names: list[str]
    ) -> dict[str, Any]:

        # Set a threshold to determine predicted classes
        threshold = 0.5

        # save model metadata
        model_metadata = {
            "timestamp": datetime.now().isoformat(),
            "label_names": label_names,
            "sample_size": {
                "training": len(tokenized_dataset['train']),
                "evaluation": len(tokenized_dataset['eval']),
            },
            "metrics_training": trainer.evaluate(),
            "metrics_labels": {},
            "samples": []
        }

        # Evaluate model
        prediction = trainer.predict(tokenized_dataset['eval']) # type: ignore
        predicted_logits = prediction.predictions
        predicted_probs = torch.sigmoid(torch.tensor(predicted_logits))

        # Initialize per label evaluation
        for label_name in label_names:
            model_metadata['metrics_labels'][label_name] = {
                "tp": 0, # True-positive: true and predicted
                "fp": 0, # False-positive: predicted but not true
                "fn": 0, # False-negative: true but not predicted
            }

        for idx, sample in enumerate(dataset['eval']):
            assert isinstance(sample, dict)
            true_labels = []
            predicted_labels = []
            for label_idx, label_name in enumerate(label_names):
                if sample['labels'][label_idx] == 1.0:
                    true_labels.append(label_name)
                if predicted_probs[idx][label_idx] > threshold:
                    predicted_labels.append(label_name)

            for label_name in true_labels:
                if label_name in predicted_labels:
                    # Expected label is predicted (true-positive)
                    model_metadata['metrics_labels'][label_name]['tp'] += 1
                else:
                    # Expected label is not predicted (false-negative)
                    model_metadata['metrics_labels'][label_name]['fn'] += 1

            for label_name in predicted_labels:
                if label_name not in true_labels:
                    # Predicted label not expected (false-positive)
                    model_metadata['metrics_labels'][label_name]['fp'] += 1

            model_metadata['samples'].append(
                {
                    "text": sample['text'],
                    "true_labels": true_labels,
                    "predicted_labels": predicted_labels
                }
            )

        # Calculate precision and recall for labels
        for label_name in label_names:
            if (model_metadata['metrics_labels'][label_name]['tp'] 
                + model_metadata['metrics_labels'][label_name]['fp']) > 0:

                model_metadata['metrics_labels'][label_name]['precision'] = \
                    (model_metadata['metrics_labels'][label_name]['tp'] / 
                        (model_metadata['metrics_labels'][label_name]['tp'] 
                            + model_metadata['metrics_labels'][label_name]['fp']))

            if (model_metadata['metrics_labels'][label_name]['tp'] 
                + model_metadata['metrics_labels'][label_name]['fn']) > 0:

                model_metadata['metrics_labels'][label_name]['recall'] = \
                    (model_metadata['metrics_labels'][label_name]['tp'] / 
                        (model_metadata['metrics_labels'][label_name]['tp'] 
                            + model_metadata['metrics_labels'][label_name]['fn']))

        return model_metadata


    def train(self) -> None:

        logging.info(f"Starting training with model: {self.base_model_name}")
        logging.info(f"Using device: {self.device}")
        
        # Load data
        labels, label_weights, ds = self.load_data()
        logging.info(f"Loaded {len(ds['train'])} training samples and {len(ds['eval'])} validation samples")
        logging.info(f"Number of labels: {len(self.label_names)}")

        # Load the tokenizer for the specified model
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
        )
        # Ensure tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or '[PAD]'
        tokenized_ds = ds.map(functools.partial(self.tokenize_examples, tokenizer=tokenizer), batched=True)
        tokenized_ds = tokenized_ds.with_format('torch')

        # Load the Hugging Face model on CPU first to avoid init_empty_weights error
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=labels.shape[1],
        )
        model = model.to(self.device)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.train()

        # define custom batch preprocessor
        def collate_fn(batch, tokenizer):
            dict_keys = ['input_ids', 'attention_mask', 'labels']
            d: dict[str, Any] = {k: [dic[k] for dic in batch] for k in dict_keys}
            d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
                d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(self.device)  # Move to device
            d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
                d['attention_mask'], batch_first=True, padding_value=0
            ).to(self.device)  # Move to device
            d['labels'] = torch.stack(d['labels']).to(self.device)  # Move to device
            return d

        # define which metrics to compute for evaluation
        def compute_metrics(p):
            predictions, labels = p
            f1_micro = f1_score(labels, predictions > 0, average = 'micro')
            f1_macro = f1_score(labels, predictions > 0, average = 'macro')
            f1_weighted = f1_score(labels, predictions > 0, average = 'weighted')
            return {
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted
            }

        # define training args
        training_args = TrainingArguments(
            output_dir=self.interim_model_path,
            learning_rate=1e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            weight_decay=0.1,   # This is quite high due to limited training data
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True
        )

        # train
        trainer = CustomTrainer(
            device=self.device,
            model=model,
            args=training_args,
            train_dataset=tokenized_ds['train'],
            eval_dataset=tokenized_ds['eval'],
            tokenizer=tokenizer,
            data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            label_weights=torch.tensor(label_weights, device=self.device)
        )

        trainer.train()

        # Save model
        logging.info(f"Saving model to {self.output_model_path}")
        trainer.save_model(self.output_model_path)
        tokenizer.save_pretrained(self.output_model_path)

        # Save model metadata
        logging.info(f"Saving model metadata to {self.output_meta_path}")
        model_metadata = self.summarize_training(trainer, ds, tokenized_ds, self.label_names)
        with open(self.output_meta_path, 'w') as fp:
            json.dump(model_metadata, fp, indent=2)
            
        logging.info("Training completed successfully!")

if __name__ == "__main__":
    trainer = QuoteClassificationTrainer()
    trainer.parse_arguments_training()
    trainer.train()
