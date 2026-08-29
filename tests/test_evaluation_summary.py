from pathlib import Path
from typing import Any

from urbanixm_content_classifier.evaluation_summary import (
    CommandLineArguments,
    EvaluationSummarizer,
)


def test_sub_label_results_include_semantics_and_split_metrics(
    tmp_path: Path,
) -> None:
    report: dict[str, Any] = {
        "evaluations": [
            {
                "label": "positive_indirect",
                "description": "Assigned through topic derivation.",
                "target": 1,
                "weight_multiplier": 0.75,
                "evaluation": {
                    split: {
                        "count": 3,
                        "accuracy": 2 / 3,
                        "mean_positive_probability": 0.61,
                    }
                    for split in ("train", "validation", "test")
                },
            }
        ]
    }
    output_path = tmp_path / "summary.md"
    summarizer = EvaluationSummarizer(CommandLineArguments(data_dir="unused"))

    with output_path.open("w") as output:
        summarizer.report_file = output
        summarizer.write_sub_label_results([(None, report)])

    summary = output_path.read_text()
    assert "### Sub-label results" in summary
    assert "- **positive_indirect**: Assigned through topic derivation." in summary
    assert summary.index("**Sub-labels**") < summary.index("**Columns**")
    table_header = next(line for line in summary.splitlines() if line.startswith("|"))
    assert "description" not in table_header
    assert "positive_indirect" in summary
    assert "positive" in summary
    assert "0.75" in summary
    assert "0.61" in summary


def test_sub_label_results_split_multiple_boolean_classifiers(
    tmp_path: Path,
) -> None:
    def report(description: str) -> dict[str, Any]:
        return {
            "evaluations": [
                {
                    "label": "positive_direct",
                    "description": description,
                    "evaluation": {
                        split: {"count": 1, "accuracy": 1.0}
                        for split in ("train", "validation", "test")
                    },
                }
            ]
        }

    output_path = tmp_path / "multiple-summary.md"
    summarizer = EvaluationSummarizer(CommandLineArguments(data_dir="unused"))

    with output_path.open("w") as output:
        summarizer.report_file = output
        summarizer.write_sub_label_results(
            [
                ("modal-share-cycling", report("Cycling topic.")),
                ("Reykjavik", report("Place topic.")),
            ]
        )

    summary = output_path.read_text()
    assert "#### modal-share-cycling" in summary
    assert "#### Reykjavik" in summary
    assert "- **positive_direct**: Cycling topic." in summary
    assert "- **positive_direct**: Place topic." in summary
    assert "classifier" not in summary
    assert summary.count("| sub-label") == 2


def test_legacy_positive_indirect_uses_historical_multiplier(tmp_path: Path) -> None:
    empty_result: dict[str, int | float] = {"count": 0, "accuracy": 0.0}
    report: dict[str, Any] = {
        "evaluations": [
            {
                "label": "positive_indirect",
                "evaluation": {
                    split: empty_result for split in ("train", "validation", "test")
                },
            }
        ]
    }
    output_path = tmp_path / "legacy-summary.md"
    summarizer = EvaluationSummarizer(CommandLineArguments(data_dir="unused"))

    with output_path.open("w") as output:
        summarizer.report_file = output
        summarizer.write_sub_label_results([(None, report)])

    summary = output_path.read_text()
    assert "0.75" in summary
    assert "Not recorded" in summary
