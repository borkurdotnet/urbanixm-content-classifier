import gzip
import json
from pathlib import Path

from urbanixm_content_classifier.article_classifier import (
    ArticleClassificationTrainer,
    ObjectiveType,
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
