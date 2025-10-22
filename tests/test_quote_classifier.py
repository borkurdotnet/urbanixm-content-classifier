import json
import os
import sys
import pytest
from unittest.mock import patch
from urbanixm_content_classifier.quote_classifier import QuoteClassificationTrainer


@pytest.fixture
def default_test_args() -> list[str]:
    """Default test arguments for quote classifier"""
    return [
        "test_script",
        "--data_dir",
        "tests/data",
        "--objective_type",
        "quote_types",
    ]


@pytest.fixture
def override_base_model_args() -> list[str]:
    """Arguments for quote classifier, overriding base model"""
    return [
        "test_script",
        "--base_model",
        "bert-base",
        "--data_dir",
        "tests/data",
        "--objective_type",
        "quote_types",
    ]


def test_parse_arguments_training(default_test_args: list[str]):
    """
    Test the argument parsing of the quote classification trainer
    """
    with patch.object(sys, "argv", default_test_args):
        trainer = QuoteClassificationTrainer()
        trainer.parse_arguments_training()

        assert trainer.base_model_name == "distilroberta-base"
        assert trainer.data_dir == "tests/data"
        assert trainer.objective_type.value == "quote_types"

        # assert trainer.label_names is None  # initiated at data load
        assert (
            trainer.training_data_path
            == "tests/data/classifiers/training-data/quotes_quote_types.csv"
        )
        assert (
            trainer.interim_model_path
            == "tests/data/classifiers/models/interim/quotes_quote_types_multilabel"
        )
        assert (
            trainer.output_model_path
            == "tests/data/classifiers/models/final/quotes_quote_types_multilabel"
        )
        assert (
            trainer.output_meta_path
            == "tests/data/classifiers/models/final/quotes_quote_types_multilabel.json"
        )


def test_parse_arguments_training_with_base_model(override_base_model_args: list[str]):
    """
    Test the argument parsing of the quote classification trainer
    """
    with patch.object(sys, "argv", override_base_model_args):
        trainer = QuoteClassificationTrainer()
        trainer.parse_arguments_training()

        assert trainer.base_model_name == "bert-base"


def test_load_text_and_labels():
    label_names, text, labels = QuoteClassificationTrainer.load_text_and_labels(
        "tests/data/classifiers/training-data/quotes_quote_types.csv"
    )

    assert label_names == [
        "impact",
        "informative",
        "infrastructure",
        "off-topic",
        "policy",
    ]

    assert len(text) == 100
    assert len(labels) == 100


def test_load_data(default_test_args: list[str]):
    with patch.object(sys, "argv", default_test_args):
        trainer = QuoteClassificationTrainer()
        trainer.parse_arguments_training()
        labels, label_weights, dataset_dict = trainer.load_data()

        assert trainer.label_names == [
            "impact",
            "informative",
            "infrastructure",
            "off-topic",
            "policy",
        ]

        assert len(labels) == 100
        assert len(label_weights) == 5
        assert len(dataset_dict) == 2
        assert "train" in dataset_dict
        assert "eval" in dataset_dict


@pytest.mark.slow
def test_train():
    """
    Test the training of a classifier

    Warning: This test can take a while (~30s) and downloads models!
    Run with: uv run pytest -m slow
    """

    # Clean existing model
    import shutil

    model_dirs = [
        "tests/data/classifiers/models/final/quotes_quote_types_multilabel",
        "tests/data/classifiers/models/interim/quotes_quote_types_multilabel",
    ]
    model_files = [
        "tests/data/classifiers/models/final/quotes_quote_types_multilabel.json"
    ]

    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

    for model_file in model_files:
        if os.path.exists(model_file):
            os.remove(model_file)

    # Mock command line arguments
    sys.argv = [
        "test_script",
        "--data_dir",
        "tests/data",
        "--objective_type",
        "quote_types",
    ]

    trainer = QuoteClassificationTrainer()
    trainer.parse_arguments_training()
    trainer.train()

    with open(model_files[0], "r") as fp:
        model_metadata = json.load(fp)

        assert len(model_metadata["label_names"]) == 5
        assert "impact" in model_metadata["label_names"]
        assert "informative" in model_metadata["label_names"]
        assert "infrastructure" in model_metadata["label_names"]
        assert "off-topic" in model_metadata["label_names"]
        assert "policy" in model_metadata["label_names"]
