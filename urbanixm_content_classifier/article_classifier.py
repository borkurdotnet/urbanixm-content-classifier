import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import gzip
import hashlib
from importlib.metadata import version
import json
import pickle
import jsonlines
import logging
import os
import platform
from pathlib import Path
import pandas as pd
import shutil
import subprocess
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.feature_extraction.text import TfidfTransformer  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import precision_recall_curve  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.utils import shuffle  # type: ignore
from typing import Any, Iterable, Iterator, cast

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

POSITIVE_TEST_COUNT_MIN = 10
DEFAULT_RANDOM_SEED = 42
METADATA_SCHEMA_VERSION = 2
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ObjectiveType(Enum):
    ON_TOPIC = "on_topic"
    QUOTABLE = "quotable"
    TOPICS = "topics"
    PLACES = "places"


@dataclass
class SubLabel:
    name: str
    description: str
    target: int
    weight_multiplier: float = 1.0
    samples: list[str] = field(default_factory=list)  # type: ignore

    def __post_init__(self) -> None:
        if self.target not in (0, 1):
            raise ValueError("Sub-label target must be 0 or 1")
        if self.weight_multiplier < 0:
            raise ValueError("Sub-label weight multiplier cannot be negative")


@dataclass
class SubLabelCounts:
    train: int
    validation: int
    test: int


def build_sub_labels(objective_type: ObjectiveType) -> list[SubLabel]:
    if objective_type == ObjectiveType.ON_TOPIC:
        return [
            SubLabel(
                name="on-topic",
                description="Article is directly labelled as related to urbanism.",
                target=1,
            ),
            SubLabel(
                name="off-topic",
                description="Article is directly labelled as unrelated to urbanism.",
                target=0,
            ),
        ]
    if objective_type == ObjectiveType.QUOTABLE:
        return [
            SubLabel(
                name="quotable",
                description="Urbanism article is directly labelled quotable.",
                target=1,
            ),
            SubLabel(
                name="unquotable",
                description="Article is about urbanism but directly labelled as unquotable.",
                target=0,
            ),
            SubLabel(
                name="off-topic",
                description="Article is unrelated to urbanism and therefore not quotable.",
                target=0,
            ),
        ]
    objective_name = "topic" if objective_type == ObjectiveType.TOPICS else "place"
    return [
        SubLabel(
            name="direct",
            description=f"Article is directly assigned the target {objective_name}.",
            target=1,
        ),
        SubLabel(
            name="derived",
            description=f"Article is assigned the target {objective_name} through hierarchy derivation.",
            target=1,
            weight_multiplier=0.75,
        ),
        SubLabel(
            name="other-topic"
            if objective_type == ObjectiveType.TOPICS
            else "other-place",
            description=f"Urbanism article is not assigned the target {objective_name}.",
            target=0,
        ),
        SubLabel(
            name="off-topic",
            description="Article is directly labelled as unrelated to urbanism.",
            target=0,
        ),
    ]


@dataclass
class DatasetCounts:
    positive_train: int
    positive_validation: int
    positive_test: int
    negative_train: int
    negative_validation: int
    negative_test: int
    sub_labels: dict[str, SubLabelCounts]

    def to_dict(self) -> dict[str, Any]:
        # Return a json serializable dict
        return {
            "positive_train": self.positive_train,
            "positive_validation": self.positive_validation,
            "positive_test": self.positive_test,
            "negative_train": self.negative_train,
            "negative_validation": self.negative_validation,
            "negative_test": self.negative_test,
            "sub_labels": {
                name: {
                    "train": counts.train,
                    "validation": counts.validation,
                    "test": counts.test,
                }
                for name, counts in self.sub_labels.items()
            },
        }


@dataclass
class Dataset:
    counts: DatasetCounts
    data_train: pd.DataFrame
    data_validation: pd.DataFrame
    data_test: pd.DataFrame


@dataclass
class EvaluationResult:
    accuracy: float
    count: int
    mean_positive_probability: float


@dataclass
class Evaluation:
    train: EvaluationResult
    validation: EvaluationResult
    test: EvaluationResult

    def to_dict(self) -> dict[str, Any]:
        # Return a json serializable dict
        return {
            "train": {
                "accuracy": self.train.accuracy,
                "count": self.train.count,
                "mean_positive_probability": self.train.mean_positive_probability,
            },
            "validation": {
                "accuracy": self.validation.accuracy,
                "count": self.validation.count,
                "mean_positive_probability": self.validation.mean_positive_probability,
            },
            "test": {
                "accuracy": self.test.accuracy,
                "count": self.test.count,
                "mean_positive_probability": self.test.mean_positive_probability,
            },
        }


@dataclass
class ClassificationMetrics:
    count: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "average_precision": self.average_precision,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_positive": self.true_positive,
        }


@dataclass
class ClassificationEvaluation:
    train: ClassificationMetrics
    validation: ClassificationMetrics
    test: ClassificationMetrics

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        return {
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
        }


@dataclass
class LabelledEvaluation:
    label: str
    evaluation: Evaluation
    description: str | None = None
    target: int | None = None
    weight_multiplier: float | None = None

    def to_dict(self) -> dict[str, Any]:
        # Return a json serializable dict
        result: dict[str, Any] = {
            "label": self.label,
            "evaluation": self.evaluation.to_dict(),
        }
        if self.description is not None:
            result["description"] = self.description
        if self.target is not None:
            result["target"] = self.target
        if self.weight_multiplier is not None:
            result["weight_multiplier"] = self.weight_multiplier
        return result


@dataclass
class EvaluatedModel:
    model: Any
    counts: DatasetCounts
    sub_labels: list[SubLabel]
    evaluations: list[LabelledEvaluation]
    decision_threshold: float
    metrics: ClassificationEvaluation
    best_parameters: dict[str, Any]
    cv_average_precision: float
    cv_folds: int
    selection_metric: str
    n_jobs: int

    def to_dict(self) -> dict[str, Any]:
        # Return a json serializable dict
        return {
            "counts": self.counts,
            "sub_labels": [
                {
                    "name": sub_label.name,
                    "description": sub_label.description,
                    "target": sub_label.target,
                    "weight_multiplier": sub_label.weight_multiplier,
                }
                for sub_label in self.sub_labels
            ],
            "evaluations": [e.to_dict() for e in self.evaluations],
            "decision_threshold": self.decision_threshold,
            "metrics": self.metrics.to_dict(),
            "best_parameters": self.best_parameters,
            "cv_average_precision": self.cv_average_precision,
            "cv_folds": self.cv_folds,
            "selection_metric": self.selection_metric,
            "n_jobs": self.n_jobs,
        }


def select_decision_threshold(
    labels: Iterable[int], probabilities: Iterable[float]
) -> float:
    label_values = list(labels)
    probability_values = list(probabilities)
    precision_result, recall_result, threshold_result = cast(
        tuple[Iterable[float], Iterable[float], Iterable[float]],
        precision_recall_curve(label_values, probability_values),
    )
    precision = list(precision_result)
    recall = list(recall_result)
    thresholds = list(threshold_result)
    if len(thresholds) == 0:
        return 0.5

    f1_scores = [
        2 * current_precision * current_recall / (current_precision + current_recall)
        if current_precision + current_recall > 0
        else 0.0
        for current_precision, current_recall in zip(precision[:-1], recall[:-1])
    ]
    return float(thresholds[f1_scores.index(max(f1_scores))])


def calculate_classification_metrics(
    labels: Iterable[int], probabilities: Iterable[float], threshold: float
) -> ClassificationMetrics:
    label_values = list(labels)
    probability_values = list(probabilities)
    predictions = [int(probability >= threshold) for probability in probability_values]
    true_negative = sum(
        label == 0 and prediction == 0
        for label, prediction in zip(label_values, predictions)
    )
    false_positive = sum(
        label == 0 and prediction == 1
        for label, prediction in zip(label_values, predictions)
    )
    false_negative = sum(
        label == 1 and prediction == 0
        for label, prediction in zip(label_values, predictions)
    )
    true_positive = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(label_values, predictions)
    )
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = true_positive / positive_count if positive_count > 0 else 0.0
    specificity = true_negative / negative_count if negative_count > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return ClassificationMetrics(
        count=len(label_values),
        accuracy=(true_positive + true_negative) / len(label_values),
        balanced_accuracy=(recall + specificity) / 2,
        precision=precision,
        recall=recall,
        f1=f1,
        average_precision=float(
            average_precision_score(label_values, probability_values)  # type: ignore
        ),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_training_data_provenance(training_data_dir: Path) -> dict[str, Any]:
    paths = sorted(
        {
            *training_data_dir.glob("articles_*.jsonl"),
            *training_data_dir.glob("articles_*.jsonl.gz"),
            *training_data_dir.glob("articles_meta.json"),
            *training_data_dir.glob("articles_*_meta.json"),
        }
    )
    if not paths:
        raise FileNotFoundError(
            f"No article training data found in {training_data_dir}"
        )

    combined_digest = hashlib.sha256()
    files: list[dict[str, str | int]] = []
    for path in paths:
        relative_path = path.relative_to(training_data_dir).as_posix()
        file_digest = sha256_file(path)
        size_bytes = path.stat().st_size
        combined_digest.update(relative_path.encode("utf-8"))
        combined_digest.update(b"\0")
        combined_digest.update(file_digest.encode("ascii"))
        combined_digest.update(b"\0")
        files.append(
            {
                "path": relative_path,
                "sha256": file_digest,
                "size_bytes": size_bytes,
            }
        )

    return {
        "fingerprint": f"sha256:{combined_digest.hexdigest()}",
        "files": files,
    }


def run_git_command(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_code_provenance() -> dict[str, str | bool]:
    try:
        git_commit = run_git_command("rev-parse", "HEAD")
        git_dirty = bool(run_git_command("status", "--porcelain"))
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit = "unknown"
        git_dirty = True
    return {"git_commit": git_commit, "git_dirty": git_dirty}


def build_runtime_provenance() -> dict[str, str]:
    lockfile_path = REPOSITORY_ROOT / "uv.lock"
    return {
        "python": platform.python_version(),
        "scikit_learn": version("scikit-learn"),
        "pandas": version("pandas"),
        "joblib": version("joblib"),
        "uv_lock_sha256": (
            f"sha256:{sha256_file(lockfile_path)}"
            if lockfile_path.exists()
            else "unavailable"
        ),
    }


class ArticleClassificationTrainer(object):
    data_dir: str
    objective_type: ObjectiveType
    training_data_dir: str
    output_model_path: str
    random_seed: int
    n_jobs: int
    provenance: dict[str, Any] | None

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED, n_jobs: int = 1) -> None:
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.provenance = None

    def parse_arguments_training(self) -> None:
        """
        Parses command line arguments and set the appropiate class variables
        """
        parser = argparse.ArgumentParser(
            description="Trainer for quote multilabel classification"
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            required=True,
            help="Path to the directory containing training data and output model",
        )
        parser.add_argument(
            "--objective_type",
            type=str,
            required=True,
            help="Objective: [on_topic, quotable, topic, place]",
        )
        parser.add_argument(
            "--random_seed",
            type=int,
            default=DEFAULT_RANDOM_SEED,
            help=f"Seed used for train/validation/test splitting (default: {DEFAULT_RANDOM_SEED})",
        )
        parser.add_argument(
            "--n_jobs",
            type=int,
            default=1,
            help="Parallel cross-validation workers; use -1 for all CPUs (default: 1)",
        )
        args = parser.parse_args()

        if args.n_jobs == 0:
            parser.error("--n_jobs cannot be zero")
        self.random_seed = args.random_seed
        self.n_jobs = args.n_jobs

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

        # Set training data directory
        self.training_data_dir = os.path.join(
            self.data_dir, "classifiers", "training-data", "articles"
        )

        # Set output model path
        self.output_model_path = os.path.join(
            self.data_dir, "classifiers", "models", "final", "article_models"
        )

    def load_data(self, objective_label: str | None) -> list[SubLabel]:
        """
        Load the training and testing data

        :param objective_label: The topic or place label, when appropriate
        """

        if (
            self.objective_type == ObjectiveType.ON_TOPIC
            or self.objective_type == ObjectiveType.QUOTABLE
        ):
            # We are training a model for estimating if a web-page is on the general topic of urbanism
            # or a model for estimating if an urbanism web-page is content rich (quotable)
            return self.load_data_unlabelled(self.iter_training_records())
        elif (
            self.objective_type == ObjectiveType.TOPICS and objective_label is not None
        ):
            # We are training a model for estimating if a web-page is on a specific urbanism topic
            return self.load_data_labelled(
                objective_label, self.iter_training_records()
            )
        elif (
            self.objective_type == ObjectiveType.PLACES and objective_label is not None
        ):
            # We are training a model for estimating if a web-page is talking about a specific plce
            return self.load_data_labelled(
                objective_label, self.iter_training_records()
            )
        raise ValueError(
            f"Unknown objective type or label: {self.objective_type}/{objective_label}"
        )

    def iter_training_records(self) -> Iterator[dict[str, Any]]:
        training_data_dir = Path(self.training_data_dir)
        content_paths: dict[str, Path] = {}

        batch_paths = [
            *training_data_dir.glob("articles_*.jsonl"),
            *training_data_dir.glob("articles_*.jsonl.gz"),
        ]
        for path in sorted(batch_paths):
            filename = path.name.removesuffix(".gz").removesuffix(".jsonl")
            if filename.endswith("_labels"):
                continue
            batch_name = filename.removesuffix("_content")
            if batch_name in content_paths:
                raise ValueError(f"Multiple content files found for batch {batch_name}")
            content_paths[batch_name] = path

        if not content_paths:
            raise FileNotFoundError(f"No article batches found in {training_data_dir}")

        seen_urls: set[str] = set()
        for batch_name, content_path in sorted(content_paths.items(), reverse=True):
            label_paths = [
                path
                for path in (
                    training_data_dir / f"{batch_name}_labels.jsonl",
                    training_data_dir / f"{batch_name}_labels.jsonl.gz",
                )
                if path.exists()
            ]
            if len(label_paths) != 1:
                raise FileNotFoundError(
                    f"Expected one labels file for {content_path.name}, "
                    f"found {len(label_paths)}"
                )

            labels_by_url: dict[str, dict[str, Any]] = {}
            with self.open_jsonlines(label_paths[0]) as label_reader:
                for labels in label_reader:
                    url = labels["url"]
                    if url in labels_by_url:
                        raise ValueError(
                            f"Duplicate URL {url!r} in {label_paths[0].name}"
                        )
                    labels_by_url[url] = labels

            batch_urls: set[str] = set()
            with self.open_jsonlines(content_path) as content_reader:
                for article in content_reader:
                    url = article["url"]
                    if url in batch_urls:
                        raise ValueError(f"Duplicate article URL {url!r}")
                    batch_urls.add(url)
                    try:
                        labels = labels_by_url.pop(url)
                    except KeyError as error:
                        raise ValueError(
                            f"No labels found for URL {url!r} in {content_path.name}"
                        ) from error
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    yield article | labels

            if labels_by_url:
                raise ValueError(
                    f"Labels without content in {label_paths[0].name}: "
                    f"{', '.join(sorted(labels_by_url))}"
                )

    @staticmethod
    def open_jsonlines(path: Path) -> jsonlines.Reader:
        if path.suffix == ".gz":
            return jsonlines.Reader(gzip.open(path, mode="rt"))
        return jsonlines.Reader(path.open())

    def load_data_unlabelled(
        self, json_objects: Iterable[dict[str, Any]]
    ) -> list[SubLabel]:
        """
        Load data for the two special classification classes 'on-topic' and 'quotable',
        which consider, respectively, any article on urbanism to be positive
        or any urbanism article that is text-rich

        :param json_objects:
        :return:
        """
        sub_labels = build_sub_labels(self.objective_type)
        samples_by_name = {
            sub_label.name: sub_label.samples for sub_label in sub_labels
        }
        for json_object in json_objects:
            text: str = json_object["content"]

            if json_object["off_topic"]:
                # Article is off topic
                samples_by_name["off-topic"].append(text)
            else:
                # Article is on topic
                if self.objective_type == ObjectiveType.QUOTABLE:
                    if json_object["un_quotable"]:
                        # Article is on topic but unquotable (archive or something like that)
                        samples_by_name["unquotable"].append(text)
                    else:
                        # Article is on topic and quotable
                        samples_by_name["quotable"].append(text)
                else:
                    # Article is on topic and quotable
                    samples_by_name["on-topic"].append(text)
        return sub_labels

    def load_data_labelled(
        self, objective_label: str, json_objects: Iterable[dict[str, Any]]
    ) -> list[SubLabel]:
        """
        Loads the data from the json_reader creates appropriate training and testing texts
        given the topic/place label being processed

        :param objective_label: The label of the topic or the place being processed
        :param json_objects: Article content merged with its labels

        :returns: Sub-label definitions populated with texts from the input data
        """
        sub_labels = build_sub_labels(self.objective_type)
        samples_by_name = {
            sub_label.name: sub_label.samples for sub_label in sub_labels
        }
        for json_object in json_objects:
            text: str = json_object["content"]
            if json_object["off_topic"]:
                # Article is not about urbanism
                samples_by_name["off-topic"].append(text)
            else:
                # Article is on topic
                if objective_label in json_object[self.objective_type.value]["direct"]:
                    # Article has been annotated directly on topic
                    samples_by_name["direct"].append(text)
                elif (
                    objective_label in json_object[self.objective_type.value]["derived"]
                ):
                    # Article has been derived to be on topic
                    samples_by_name["derived"].append(text)
                else:
                    # Article is about urbanism but not on the desired objective label (topic or place)
                    if self.objective_type == ObjectiveType.TOPICS:
                        samples_by_name["other-topic"].append(text)
                    else:
                        samples_by_name["other-place"].append(text)
        return sub_labels

    def get_data_spit(self, sub_labels: list[SubLabel]) -> Dataset:
        """
        Split the task's training data in to train/validation/test sets

        :param sub_labels: semantic sample groups used for training and testing

        :return: Dataset with text, label, weight and sub_label columns

        where:
            - __text__ is the text of an article
            - __label__ is 0 if negative and 1 if positive
            - __weight__ combines class balancing with the sub-label multiplier
            - __sub_label__ identifies the semantic sample group
        """

        sub_label_names = [sub_label.name for sub_label in sub_labels]
        if len(sub_label_names) != len(set(sub_label_names)):
            raise ValueError("Sub-label names must be unique")

        # Reserve roughly 10% of positive cases for each holdout set.
        positive_count = sum(
            len(sub_label.samples) for sub_label in sub_labels if sub_label.target == 1
        )
        negative_count = sum(
            len(sub_label.samples) for sub_label in sub_labels if sub_label.target == 0
        )
        if positive_count == 0 or negative_count == 0:
            raise ValueError(
                "Article classification requires both positive and negative examples"
            )
        pos_holdout_count = min(
            max(POSITIVE_TEST_COUNT_MIN, int(0.1 * positive_count)),
            max((positive_count - 1) // 2, 0),
        )
        pos_neg_ratio = positive_count / negative_count

        sub_label_frames: list[DataFrame] = []
        for sub_label in sub_labels:
            frame = DataFrame(data=sub_label.samples, columns=["text"])
            frame["label"] = sub_label.target
            class_weight = 1.0 if sub_label.target == 1 else pos_neg_ratio
            frame["weight"] = class_weight * sub_label.weight_multiplier
            frame["sub_label"] = sub_label.name
            sub_label_frames.append(frame)

        # Positive train/validation/test split
        pos_train_df = pd.concat(
            [
                frame
                for frame, sub_label in zip(sub_label_frames, sub_labels)
                if sub_label.target == 1
            ],
            ignore_index=True,
        )
        pos_validation_df = pos_train_df.sample(  # type: ignore
            n=pos_holdout_count, random_state=self.random_seed
        )
        pos_train_df = pos_train_df.drop(pos_validation_df.index)
        pos_test_df = pos_train_df.sample(  # type: ignore
            n=pos_holdout_count, random_state=self.random_seed
        )
        pos_train_df = pos_train_df.drop(pos_test_df.index)

        # Negative train/validation/test split
        neg_train_df = pd.concat(
            [
                frame
                for frame, sub_label in zip(sub_label_frames, sub_labels)
                if sub_label.target == 0
            ],
            ignore_index=True,
        )
        holdout_fraction = pos_holdout_count / positive_count
        neg_holdout_count = min(
            round(holdout_fraction * neg_train_df.shape[0]),
            max((neg_train_df.shape[0] - 1) // 2, 0),
        )
        neg_validation_df = neg_train_df.sample(  # type: ignore
            n=neg_holdout_count, random_state=self.random_seed
        )
        neg_train_df = neg_train_df.drop(neg_validation_df.index)
        neg_test_df = neg_train_df.sample(  # type: ignore
            n=neg_holdout_count, random_state=self.random_seed
        )
        neg_train_df = neg_train_df.drop(neg_test_df.index)

        dataset = Dataset(
            counts=DatasetCounts(
                positive_train=pos_train_df.shape[0],
                positive_validation=pos_validation_df.shape[0],
                positive_test=pos_test_df.shape[0],
                negative_train=neg_train_df.shape[0],
                negative_validation=neg_validation_df.shape[0],
                negative_test=neg_test_df.shape[0],
                sub_labels={
                    sub_label.name: SubLabelCounts(
                        train=int(
                            (pos_train_df["sub_label"] == sub_label.name).sum()
                            if sub_label.target == 1
                            else (neg_train_df["sub_label"] == sub_label.name).sum()
                        ),
                        validation=int(
                            (pos_validation_df["sub_label"] == sub_label.name).sum()
                            if sub_label.target == 1
                            else (
                                neg_validation_df["sub_label"] == sub_label.name
                            ).sum()
                        ),
                        test=int(
                            (pos_test_df["sub_label"] == sub_label.name).sum()
                            if sub_label.target == 1
                            else (neg_test_df["sub_label"] == sub_label.name).sum()
                        ),
                    )
                    for sub_label in sub_labels
                },
            ),
            data_train=pd.DataFrame(
                shuffle(
                    pd.concat([pos_train_df, neg_train_df]),
                    random_state=self.random_seed,
                )
            ),
            data_validation=pd.DataFrame(
                shuffle(
                    pd.concat([pos_validation_df, neg_validation_df]),
                    random_state=self.random_seed,
                )
            ),
            data_test=pd.DataFrame(
                shuffle(
                    pd.concat([pos_test_df, neg_test_df]),
                    random_state=self.random_seed,
                )
            ),
        )

        return dataset

    def train_model(self, sub_labels: list[SubLabel]) -> EvaluatedModel:
        """
        Takes a set of text objects and trains a model.
        The model is evaluated along several metrics to get a complete understanding of the model performance.
        """

        training_data = self.get_data_spit(sub_labels)

        parameters: dict[str, Any] = {
            "vect__ngram_range": [(1, 1), (1, 2)],
            "vect__min_df": [1, 3],
            "clf__C": [0.1, 1.0, 10.0],
        }

        cv_folds = min(
            5,
            training_data.counts.positive_train,
            training_data.counts.negative_train,
        )
        if cv_folds < 3:
            raise ValueError(
                "At least three training examples per class are required for "
                "hyperparameter selection"
            )
        cross_validation = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_seed,
        )

        text_clf = Pipeline(
            [
                ("vect", CountVectorizer(stop_words="english")),
                ("tfidf", TfidfTransformer()),
                (
                    "clf",
                    SVC(
                        kernel="linear",
                        probability=True,
                        random_state=self.random_seed,
                    ),
                ),
            ]
        )
        gs_clf = GridSearchCV(  # type: ignore
            text_clf,
            parameters,
            scoring="average_precision",
            cv=cross_validation,
            n_jobs=self.n_jobs,
            pre_dispatch="n_jobs",
            refit=True,
        )

        gs_clf = gs_clf.fit(  # type: ignore
            list(training_data.data_train["text"]),
            training_data.data_train["label"],
            clf__sample_weight=training_data.data_train["weight"],
        )

        split_probabilities = {
            "train": self.get_positive_probabilities(gs_clf, training_data.data_train),
            "validation": self.get_positive_probabilities(
                gs_clf, training_data.data_validation
            ),
            "test": self.get_positive_probabilities(gs_clf, training_data.data_test),
        }
        decision_threshold = select_decision_threshold(
            training_data.data_validation["label"],
            split_probabilities["validation"],
        )
        classification_evaluation = ClassificationEvaluation(
            train=calculate_classification_metrics(
                training_data.data_train["label"],
                split_probabilities["train"],
                decision_threshold,
            ),
            validation=calculate_classification_metrics(
                training_data.data_validation["label"],
                split_probabilities["validation"],
                decision_threshold,
            ),
            test=calculate_classification_metrics(
                training_data.data_test["label"],
                split_probabilities["test"],
                decision_threshold,
            ),
        )

        collected_evaluation: list[LabelledEvaluation] = []

        # Overall accuracy
        overall_evaluation = LabelledEvaluation(
            label="overall",
            evaluation=Evaluation(
                train=EvaluationResult(
                    accuracy=classification_evaluation.train.accuracy,
                    count=training_data.data_train.shape[0],
                    mean_positive_probability=float(
                        sum(split_probabilities["train"])
                        / len(split_probabilities["train"])
                    ),
                ),
                validation=EvaluationResult(
                    accuracy=classification_evaluation.validation.accuracy,
                    count=training_data.data_validation.shape[0],
                    mean_positive_probability=float(
                        sum(split_probabilities["validation"])
                        / len(split_probabilities["validation"])
                    ),
                ),
                test=EvaluationResult(
                    accuracy=classification_evaluation.test.accuracy,
                    count=training_data.data_test.shape[0],
                    mean_positive_probability=float(
                        sum(split_probabilities["test"])
                        / len(split_probabilities["test"])
                    ),
                ),
            ),
        )
        collected_evaluation.append(overall_evaluation)

        for sub_label in sub_labels:
            split_data = {
                "train": training_data.data_train[
                    training_data.data_train["sub_label"] == sub_label.name
                ],
                "validation": training_data.data_validation[
                    training_data.data_validation["sub_label"] == sub_label.name
                ],
                "test": training_data.data_test[
                    training_data.data_test["sub_label"] == sub_label.name
                ],
            }

            def evaluate_split(split: str) -> EvaluationResult:
                data = split_data[split]
                probabilities = (
                    self.get_positive_probabilities(gs_clf, data)
                    if data.shape[0] > 0
                    else []
                )
                return EvaluationResult(
                    accuracy=self.score_at_threshold(gs_clf, data, decision_threshold)
                    if data.shape[0] > 0
                    else 0.0,
                    count=data.shape[0],
                    mean_positive_probability=(
                        float(sum(probabilities) / len(probabilities))
                        if probabilities
                        else 0.0
                    ),
                )

            collected_evaluation.append(
                LabelledEvaluation(
                    label=sub_label.name,
                    description=sub_label.description,
                    target=sub_label.target,
                    weight_multiplier=sub_label.weight_multiplier,
                    evaluation=Evaluation(
                        train=evaluate_split("train"),
                        validation=evaluate_split("validation"),
                        test=evaluate_split("test"),
                    ),
                )
            )

        return EvaluatedModel(
            model=gs_clf,
            counts=training_data.counts,
            sub_labels=sub_labels,
            evaluations=collected_evaluation,
            decision_threshold=decision_threshold,
            metrics=classification_evaluation,
            best_parameters={
                parameter: list(cast(tuple[object, ...], value))
                if isinstance(value, tuple)
                else value
                for parameter, value in gs_clf.best_params_.items()  # type: ignore
            },
            cv_average_precision=float(gs_clf.best_score_),  # type: ignore
            cv_folds=cv_folds,
            selection_metric="average_precision",
            n_jobs=self.n_jobs,
        )

    @staticmethod
    def get_positive_probabilities(model: Any, data: DataFrame) -> list[float]:
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(list(data["text"]))[:, positive_index]
        return [float(probability) for probability in probabilities]

    def score_at_threshold(
        self, model: Any, data: DataFrame, decision_threshold: float
    ) -> float:
        probabilities = self.get_positive_probabilities(model, data)
        predictions = [
            int(probability >= decision_threshold) for probability in probabilities
        ]
        correct_count = sum(
            label == prediction for label, prediction in zip(data["label"], predictions)
        )
        return correct_count / len(predictions)

    def save_classifier_model(
        self, objective_label: str | None, evaluated_model: EvaluatedModel
    ) -> None:
        """
        Save classifier model and its evaluation meta-data
        :param objective_label: For topics and places, the topic or place label
        :param evaluated_model: An object containing both the model and the meta-data
        """

        if self.provenance is None:
            self.provenance = {
                "training_data": build_training_data_provenance(
                    Path(self.training_data_dir)
                ),
                "code": build_code_provenance(),
                "runtime": build_runtime_provenance(),
            }

        # Save classifier meta-data
        metadata: dict[str, Any] = {
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "objective_type": self.objective_type.value,
            "objective_label": objective_label,
            **self.provenance,
            "random_seed": self.random_seed,
            "decision_threshold": evaluated_model.decision_threshold,
            "best_parameters": evaluated_model.best_parameters,
            "cv_average_precision": evaluated_model.cv_average_precision,
            "cv_folds": evaluated_model.cv_folds,
            "selection_metric": evaluated_model.selection_metric,
            "n_jobs": evaluated_model.n_jobs,
            "counts": evaluated_model.counts.to_dict(),
            "sub_labels": [
                {
                    "name": sub_label.name,
                    "description": sub_label.description,
                    "target": sub_label.target,
                    "weight_multiplier": sub_label.weight_multiplier,
                    "counts": {
                        "train": evaluated_model.counts.sub_labels[
                            sub_label.name
                        ].train,
                        "validation": evaluated_model.counts.sub_labels[
                            sub_label.name
                        ].validation,
                        "test": evaluated_model.counts.sub_labels[sub_label.name].test,
                    },
                }
                for sub_label in evaluated_model.sub_labels
            ],
            "evaluations": [e.to_dict() for e in evaluated_model.evaluations],
            "metrics": evaluated_model.metrics.to_dict(),
        }
        model_meta_filename = self.objective_type.value
        if objective_label is not None:
            model_meta_filename += f"_{objective_label}"
        model_meta_filename += ".json"
        model_meta_path = os.path.join(
            self.output_model_path, self.objective_type.value
        )
        os.makedirs(model_meta_path, exist_ok=True)
        model_meta_path = os.path.join(model_meta_path, model_meta_filename)

        with open(model_meta_path, "w") as fout:
            fout.write(json.dumps(metadata, indent=2))
            fout.close()

        # Save classifier model pickle
        model_pickle_path = model_meta_path.replace(".json", ".pickle")
        with open(model_pickle_path, "wb") as fh:
            pickle.dump(evaluated_model.model, fh)
            fh.close()

    def clear_objective_models(self) -> None:
        objective_model_path = os.path.join(
            self.output_model_path, self.objective_type.value
        )
        if os.path.isdir(objective_model_path):
            logging.info("Removing stale models from %s", objective_model_path)
            shutil.rmtree(objective_model_path)

    def get_objective_label_stats(self) -> dict[str, dict[str, int]]:
        label_stats: dict[str, dict[str, int]] = {}
        for record in self.iter_training_records():
            if record["off_topic"]:
                continue
            for source in ("direct", "derived"):
                for label in record[self.objective_type.value][source]:
                    stats = label_stats.setdefault(label, {"direct": 0, "derived": 0})
                    stats[source] += 1
        return label_stats

    def train_models(self) -> None:
        """
        Trains the appropriate models using information passed on the command line
        """

        self.clear_objective_models()

        # Global models
        if (
            self.objective_type == ObjectiveType.ON_TOPIC
            or self.objective_type == ObjectiveType.QUOTABLE
        ):
            # Train a single model for either the on-topic class or quotable class
            sub_labels = self.load_data(objective_label=None)
            evaluated_model = self.train_model(sub_labels)
            self.save_classifier_model(
                objective_label=None, evaluated_model=evaluated_model
            )

            return

        label_stats = self.get_objective_label_stats()

        if self.objective_type == ObjectiveType.TOPICS:
            # Train a classifier for each topic with sufficient training data
            for topic_label, topic_stats in label_stats.items():
                # Train a classifier for topic with label topic_label
                training_instances = topic_stats["direct"] + topic_stats["derived"]
                logging.info(
                    f"Building model for {topic_label} ({training_instances} instances)"
                )
                if training_instances < 3 * POSITIVE_TEST_COUNT_MIN:
                    logging.warning(
                        "Topic has insufficient training data: {}".format(topic_label)
                    )
                    continue

                sub_labels = self.load_data(objective_label=topic_label)
                evaluated_model = self.train_model(sub_labels)
                self.save_classifier_model(
                    objective_label=topic_label, evaluated_model=evaluated_model
                )

        if self.objective_type == ObjectiveType.PLACES:
            # Train a classifier for each place with sufficient training data
            for place_label, place_stats in label_stats.items():
                # Train a classifier for place with label place_label
                training_instances = place_stats["direct"] + place_stats["derived"]
                logging.info(
                    f"Building model for {place_label} ({training_instances} instances)"
                )
                if training_instances < 3 * POSITIVE_TEST_COUNT_MIN:
                    logging.warning(
                        "Place has insufficient training data: {}".format(place_label)
                    )
                    continue

                sub_labels = self.load_data(objective_label=place_label)
                evaluated_model = self.train_model(sub_labels)
                self.save_classifier_model(
                    objective_label=place_label, evaluated_model=evaluated_model
                )


if __name__ == "__main__":
    """
    Parses the arguments passed on the command line and trains the appropriate models
    """
    trainer = ArticleClassificationTrainer()
    trainer.parse_arguments_training()
    trainer.train_models()
