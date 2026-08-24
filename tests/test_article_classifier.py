import gzip
import json
from pathlib import Path

from urbanixm_content_classifier.article_classifier import (
    ArticleClassificationTrainer,
    ObjectiveType,
    Texts,
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
