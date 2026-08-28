import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from urbanixm_content_classifier.article_classifier import (
    ArticleClassificationTrainer,
    ObjectiveType,
    Texts,
    build_training_data_provenance,
    calculate_classification_metrics,
    select_decision_threshold,
)


def write_jsonlines(path: Path, records: list[dict[str, object]]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")


def test_load_data_uses_latest_timestamped_record_for_duplicate_urls(
    tmp_path: Path,
) -> None:
    write_jsonlines(
        tmp_path / "articles_20250716104400_content.jsonl.gz",
        [
            {"url": "https://example.com/duplicate", "content": "older"},
            {"url": "https://example.com/older-only", "content": "older only"},
        ],
    )
    write_jsonlines(
        tmp_path / "articles_20250716104400_labels.jsonl.gz",
        [
            {
                "url": "https://example.com/older-only",
                "off_topic": False,
                "topics": {"direct": ["housing"], "derived": []},
            },
            {"url": "https://example.com/duplicate", "off_topic": True},
        ],
    )
    write_jsonlines(
        tmp_path / "articles_20260826170325_content.jsonl.gz",
        [
            {"url": "https://example.com/newer-only", "content": "newer only"},
            {"url": "https://example.com/duplicate", "content": "newer"},
        ],
    )
    write_jsonlines(
        tmp_path / "articles_20260826170325_labels.jsonl.gz",
        [
            {
                "url": "https://example.com/duplicate",
                "off_topic": False,
                "topics": {"direct": ["mobility"], "derived": []},
            },
            {"url": "https://example.com/newer-only", "off_topic": True},
        ],
    )

    trainer = ArticleClassificationTrainer()
    trainer.training_data_dir = str(tmp_path)
    trainer.objective_type = ObjectiveType.ON_TOPIC

    texts = trainer.load_data(objective_label=None)

    assert texts.positive_direct == ["newer", "older only"]
    assert texts.negative_direct == ["newer only"]

    trainer.objective_type = ObjectiveType.TOPICS
    assert trainer.get_objective_label_stats() == {
        "mobility": {"direct": 1, "derived": 0},
        "housing": {"direct": 1, "derived": 0},
    }


def test_iter_training_records_rejects_duplicate_article_urls(tmp_path: Path) -> None:
    write_jsonlines(
        tmp_path / "articles_0001.jsonl",
        [
            {"url": "https://example.com/duplicate", "content": "first"},
            {"url": "https://example.com/duplicate", "content": "second"},
        ],
    )
    write_jsonlines(
        tmp_path / "articles_0001_labels.jsonl",
        [{"url": "https://example.com/duplicate", "off_topic": False}],
    )
    trainer = ArticleClassificationTrainer()
    trainer.training_data_dir = str(tmp_path)

    with pytest.raises(ValueError, match="Duplicate article URL"):
        list(trainer.iter_training_records())


def test_iter_training_records_rejects_missing_and_orphan_labels(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    write_jsonlines(
        missing_dir / "articles_0001.jsonl",
        [{"url": "https://example.com/article", "content": "article"}],
    )
    write_jsonlines(missing_dir / "articles_0001_labels.jsonl", [])
    trainer = ArticleClassificationTrainer()
    trainer.training_data_dir = str(missing_dir)

    with pytest.raises(ValueError, match="No labels found"):
        list(trainer.iter_training_records())

    orphan_dir = tmp_path / "orphan"
    orphan_dir.mkdir()
    write_jsonlines(
        orphan_dir / "articles_0001.jsonl",
        [{"url": "https://example.com/article", "content": "article"}],
    )
    write_jsonlines(
        orphan_dir / "articles_0001_labels.jsonl",
        [
            {"url": "https://example.com/article", "off_topic": False},
            {"url": "https://example.com/orphan", "off_topic": False},
        ],
    )
    trainer.training_data_dir = str(orphan_dir)

    with pytest.raises(ValueError, match="Labels without content"):
        list(trainer.iter_training_records())


def test_label_sources_determine_classes_and_sample_weights() -> None:
    trainer = ArticleClassificationTrainer()
    trainer.objective_type = ObjectiveType.TOPICS
    records: list[dict[str, Any]] = [
        {
            "content": "direct",
            "off_topic": False,
            "topics": {"direct": ["target"], "derived": []},
        },
        {
            "content": "derived",
            "off_topic": False,
            "topics": {"direct": [], "derived": ["target"]},
        },
        {
            "content": "other topic",
            "off_topic": False,
            "topics": {"direct": ["other"], "derived": []},
        },
        {
            "content": "off topic",
            "off_topic": True,
            "topics": {"direct": [], "derived": []},
        },
    ]

    texts = trainer.load_data_labelled("target", records)

    assert texts.positive_direct == ["direct"]
    assert texts.positive_indirect == ["derived"]
    assert texts.negative_indirect == ["other topic"]
    assert texts.negative_direct == ["off topic"]

    weighted_texts = Texts(
        positive_direct=[f"positive-direct-{index}" for index in range(20)],
        positive_indirect=[f"positive-indirect-{index}" for index in range(20)],
        negative_direct=[f"negative-direct-{index}" for index in range(20)],
        negative_indirect=[f"negative-indirect-{index}" for index in range(20)],
    )
    dataset = trainer.get_data_spit(weighted_texts)
    all_data = pd.concat(
        [dataset.data_train, dataset.data_validation, dataset.data_test]
    )

    assert set(all_data.loc[all_data["d/i"] == "direct", "weight"]) == {1.0}
    assert set(
        all_data.loc[
            (all_data["p/n"] == "positive") & (all_data["d/i"] == "indirect"),
            "weight",
        ]
    ) == {0.75}
    assert set(all_data.loc[all_data["p/n"] == "negative", "weight"]) == {1.0}


def test_data_split_rejects_single_class_dataset() -> None:
    trainer = ArticleClassificationTrainer()

    with pytest.raises(ValueError, match="positive and negative"):
        trainer.get_data_spit(Texts(positive_direct=["positive"] * 20))


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


def test_train_model_selects_threshold_and_evaluates_test_set(
    tmp_path: Path,
) -> None:
    trainer = ArticleClassificationTrainer(random_seed=123)
    trainer.output_model_path = str(tmp_path)
    trainer.training_data_dir = str(tmp_path / "training")
    trainer.objective_type = ObjectiveType.ON_TOPIC
    training_data_dir = Path(trainer.training_data_dir)
    training_data_dir.mkdir()
    (training_data_dir / "articles_0001.jsonl").write_text('{"content": "one"}\n')
    (training_data_dir / "articles_0001_labels.jsonl").write_text(
        '{"off_topic": false}\n'
    )
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
    assert evaluated_model.selection_metric == "average_precision"
    assert evaluated_model.cv_folds == 5
    assert evaluated_model.cv_average_precision == 1.0
    assert evaluated_model.n_jobs == 1
    assert evaluated_model.model.n_jobs == 1
    assert set(evaluated_model.best_parameters) == {
        "clf__C",
        "vect__min_df",
        "vect__ngram_range",
    }

    trainer.save_classifier_model(objective_label=None, evaluated_model=evaluated_model)
    metadata = json.loads((tmp_path / "on_topic" / "on_topic.json").read_text())

    assert metadata["best_parameters"] == evaluated_model.best_parameters
    assert metadata["cv_average_precision"] == 1.0
    assert metadata["cv_folds"] == 5
    assert metadata["selection_metric"] == "average_precision"
    assert metadata["n_jobs"] == 1
    assert metadata["metadata_schema_version"] == 1
    assert metadata["created_at"].endswith("Z")
    assert metadata["training_data"]["fingerprint"].startswith("sha256:")
    assert metadata["code"]["git_commit"]
    assert isinstance(metadata["code"]["git_dirty"], bool)
    assert metadata["runtime"]["python"]
    assert metadata["runtime"]["scikit_learn"]
    assert metadata["runtime"]["uv_lock_sha256"].startswith("sha256:")


def test_saved_model_round_trip_preserves_probabilities_and_threshold(
    tmp_path: Path,
) -> None:
    trainer = ArticleClassificationTrainer(random_seed=123)
    trainer.output_model_path = str(tmp_path)
    trainer.training_data_dir = str(tmp_path / "training")
    trainer.objective_type = ObjectiveType.ON_TOPIC
    training_data_dir = Path(trainer.training_data_dir)
    training_data_dir.mkdir()
    (training_data_dir / "articles_0001.jsonl").write_text("article\n")
    texts = Texts(
        positive_direct=[f"urban cycling policy {index}" for index in range(50)],
        negative_direct=[f"celebrity fashion news {index}" for index in range(50)],
    )
    evaluated_model = trainer.train_model(texts)
    trainer.save_classifier_model(None, evaluated_model)
    sample_texts = ["urban cycling infrastructure", "celebrity fashion"]
    expected_probabilities = evaluated_model.model.predict_proba(sample_texts)
    positive_index = list(evaluated_model.model.classes_).index(1)
    expected_predictions = [
        int(probability >= evaluated_model.decision_threshold)
        for probability in expected_probabilities[:, positive_index]
    ]

    with (tmp_path / "on_topic" / "on_topic.pickle").open("rb") as model_file:
        loaded_model = pickle.load(model_file)
    metadata = json.loads((tmp_path / "on_topic" / "on_topic.json").read_text())
    loaded_probabilities = loaded_model.predict_proba(sample_texts)
    loaded_positive_index = list(loaded_model.classes_).index(1)
    predictions = [
        int(probability >= metadata["decision_threshold"])
        for probability in loaded_probabilities[:, loaded_positive_index]
    ]

    assert loaded_probabilities.tolist() == expected_probabilities.tolist()
    assert metadata["decision_threshold"] == evaluated_model.decision_threshold
    assert predictions == expected_predictions


def test_training_data_fingerprint_is_deterministic_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    for directory in (first_dir, second_dir):
        (directory / "articles_0001.jsonl").write_text("article\n")
        (directory / "articles_0001_labels.jsonl").write_text("labels\n")

    first = build_training_data_provenance(first_dir)
    second = build_training_data_provenance(second_dir)

    assert first == second
    assert [file["path"] for file in first["files"]] == [
        "articles_0001.jsonl",
        "articles_0001_labels.jsonl",
    ]

    (second_dir / "articles_0001_labels.jsonl").write_text("changed labels\n")

    assert (
        build_training_data_provenance(second_dir)["fingerprint"]
        != first["fingerprint"]
    )


def test_clear_objective_models_only_removes_selected_objective(
    tmp_path: Path,
) -> None:
    trainer = ArticleClassificationTrainer()
    trainer.output_model_path = str(tmp_path)
    trainer.objective_type = ObjectiveType.TOPICS
    topics_dir = tmp_path / "topics"
    places_dir = tmp_path / "places"
    topics_dir.mkdir()
    places_dir.mkdir()
    (topics_dir / "obsolete.json").write_text("{}")
    (places_dir / "current.json").write_text("{}")

    trainer.clear_objective_models()

    assert not topics_dir.exists()
    assert (places_dir / "current.json").exists()


def test_train_models_writes_only_eligible_current_objective_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = ArticleClassificationTrainer()
    trainer.objective_type = ObjectiveType.TOPICS
    trainer.output_model_path = str(tmp_path / "models")
    objective_dir = Path(trainer.output_model_path) / "topics"
    objective_dir.mkdir(parents=True)
    (objective_dir / "obsolete.json").write_text("{}")

    def load_data(objective_label: str | None) -> Texts:
        return Texts()

    def train_model(_texts: Texts) -> Any:
        return object()

    monkeypatch.setattr(trainer, "load_data", load_data)
    monkeypatch.setattr(trainer, "train_model", train_model)
    monkeypatch.setattr(
        trainer,
        "get_objective_label_stats",
        lambda: {
            "eligible": {"direct": 30, "derived": 0},
            "insufficient": {"direct": 29, "derived": 0},
        },
    )

    def save_artifact(objective_label: str | None, evaluated_model: Any) -> None:
        objective_dir.mkdir(parents=True, exist_ok=True)
        (objective_dir / f"{objective_label}.json").write_text("{}")

    monkeypatch.setattr(trainer, "save_classifier_model", save_artifact)

    trainer.train_models()

    assert sorted(path.name for path in objective_dir.iterdir()) == ["eligible.json"]
