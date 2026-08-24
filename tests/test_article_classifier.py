import gzip
import json
from pathlib import Path

from urbanixm_content_classifier.article_classifier import (
    ArticleClassificationTrainer,
    ObjectiveType,
    Texts,
    calculate_classification_metrics,
    select_decision_threshold,
)


def write_jsonlines(path: Path, records: list[dict[str, object]]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")


def test_load_data_joins_labels_by_url_across_batches(tmp_path: Path) -> None:
    write_jsonlines(
        tmp_path / "articles_0001.jsonl.gz",
        [
            {"url": "https://example.com/one", "content": "first"},
            {"url": "https://example.com/two", "content": "second"},
        ],
    )
    write_jsonlines(
        tmp_path / "articles_0001_labels.jsonl",
        [
            {"url": "https://example.com/two", "off_topic": True},
            {"url": "https://example.com/one", "off_topic": False},
        ],
    )
    write_jsonlines(
        tmp_path / "articles_0002.jsonl",
        [
            {"url": "https://example.com/three", "content": "third"},
            {"url": "https://example.com/four", "content": "fourth"},
        ],
    )
    write_jsonlines(
        tmp_path / "articles_0002_labels.jsonl",
        [
            {"url": "https://example.com/four", "off_topic": True},
            {"url": "https://example.com/three", "off_topic": False},
        ],
    )

    trainer = ArticleClassificationTrainer()
    trainer.training_data_dir = str(tmp_path)
    trainer.objective_type = ObjectiveType.ON_TOPIC

    texts = trainer.load_data(objective_label=None)

    assert texts.positive_direct == ["first", "third"]
    assert texts.negative_direct == ["second", "fourth"]


def test_get_data_split_includes_validation_set() -> None:
    trainer = ArticleClassificationTrainer()
    texts = Texts(
        positive_direct=[f"positive-{index}" for index in range(100)],
        negative_direct=[f"negative-{index}" for index in range(100)],
    )

    dataset = trainer.get_data_spit(texts)

    assert dataset.counts.positive_train == 80
    assert dataset.counts.positive_validation == 10
    assert dataset.counts.positive_test == 10
    assert dataset.counts.negative_train == 80
    assert dataset.counts.negative_validation == 10
    assert dataset.counts.negative_test == 10
    assert len(dataset.data_train) == 160
    assert len(dataset.data_validation) == 20
    assert len(dataset.data_test) == 20

    split_texts = [
        set(dataset.data_train["text"]),
        set(dataset.data_validation["text"]),
        set(dataset.data_test["text"]),
    ]
    assert split_texts[0].isdisjoint(split_texts[1])
    assert split_texts[0].isdisjoint(split_texts[2])
    assert split_texts[1].isdisjoint(split_texts[2])
    assert set.union(*split_texts) == {  # type: ignore
        *(f"positive-{index}" for index in range(100)),
        *(f"negative-{index}" for index in range(100)),
    }


def test_get_data_split_is_reproducible() -> None:
    trainer = ArticleClassificationTrainer(random_seed=123)
    texts = Texts(
        positive_direct=[f"positive-{index}" for index in range(100)],
        negative_direct=[f"negative-{index}" for index in range(100)],
    )

    first_dataset = trainer.get_data_spit(texts)
    second_dataset = trainer.get_data_spit(texts)

    assert first_dataset.data_train.equals(second_dataset.data_train)
    assert first_dataset.data_validation.equals(second_dataset.data_validation)
    assert first_dataset.data_test.equals(second_dataset.data_test)


def test_get_data_split_preserves_class_prevalence() -> None:
    trainer = ArticleClassificationTrainer()
    texts = Texts(
        positive_direct=[f"positive-{index}" for index in range(100)],
        negative_direct=[f"negative-{index}" for index in range(1000)],
    )

    dataset = trainer.get_data_spit(texts)

    assert dataset.counts.positive_validation == 10
    assert dataset.counts.negative_validation == 100
    assert dataset.counts.positive_test == 10
    assert dataset.counts.negative_test == 100


def test_select_decision_threshold_maximizes_validation_f1() -> None:
    labels = [1, 1, 0, 0]
    probabilities = [0.45, 0.40, 0.35, 0.10]

    threshold = select_decision_threshold(labels, probabilities)

    assert threshold == 0.40


def test_metrics_expose_all_negative_classifier() -> None:
    labels = [1] * 10 + [0] * 90
    probabilities = [0.1] * 100

    metrics = calculate_classification_metrics(labels, probabilities, threshold=0.5)

    assert metrics.accuracy == 0.9
    assert metrics.balanced_accuracy == 0.5
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.false_negative == 10
    assert metrics.true_negative == 90


def test_train_model_selects_threshold_and_evaluates_test_set() -> None:
    trainer = ArticleClassificationTrainer(random_seed=123)
    texts = Texts(
        positive_direct=[
            f"urban cycling policy infrastructure example {index}"
            for index in range(50)
        ],
        negative_direct=[
            f"celebrity fashion entertainment example {index}" for index in range(50)
        ],
    )

    evaluated_model = trainer.train_model(texts)

    assert 0.0 <= evaluated_model.decision_threshold <= 1.0
    assert evaluated_model.metrics.validation.count == 20
    assert evaluated_model.metrics.test.count == 20
    assert evaluated_model.metrics.test.true_positive == 10
    assert evaluated_model.metrics.test.true_negative == 10
