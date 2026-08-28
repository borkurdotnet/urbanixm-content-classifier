import argparse
from dataclasses import dataclass
from io import TextIOWrapper
import json
import logging
import os
import pandas as pd

from typing import Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class CommandLineArguments:
    data_dir: str


@dataclass
class ReportSection:
    type: str
    title: str
    model_name: str | None = None
    models_dir: str | None = None


@dataclass
class EvaluationReport:
    path: str
    sections: list[ReportSection]


class EvaluationSummarizer(object):
    cl_arguments: CommandLineArguments
    report_file: TextIOWrapper

    def __init__(self, args: CommandLineArguments) -> None:
        self.cl_arguments = args

    def write_column_descriptions(self, descriptions: dict[str, str]) -> None:
        self.report_file.write("**Columns**\n\n")
        for column, description in descriptions.items():
            self.report_file.write(f"- **{column}**: {description}\n")
        self.report_file.write("\n")

    def generate_summary(self, report_conf: EvaluationReport) -> None:
        self.report_file = open(report_conf.path, "w")

        self.report_file.write("# Urbanixm Content Classifier Evaluation Report\n\n")

        for section in report_conf.sections:
            self.report_file.write(f"## {section.title}\n\n")

            if section.type == "single-boolean":  # FIXME: get from enum
                self.single_boolean_classifier_summary(section_conf=section)
            elif section.type == "multiple-boolean":  # FIXME: get from enum
                self.multiple_boolean_classifier_summary(section_conf=section)
            elif section.type == "multilabel":  # FIXME: get from enum
                self.multiclass_classifier_summary(section_conf=section)
        self.report_file.close()

    def single_boolean_classifier_summary(self, section_conf: ReportSection) -> None:
        if section_conf.model_name is not None:
            with open(
                os.path.join(
                    self.cl_arguments.data_dir,
                    "classifiers",
                    "models",
                    "final",
                    "article_models",
                    section_conf.model_name,
                    f"{section_conf.model_name}.json",
                )
            ) as fp:
                evaluation_report = json.load(fp)
                fp.close()

            if "metrics" in evaluation_report:
                test_metrics = evaluation_report["metrics"]["test"]
                self.report_file.write(
                    f"Decision threshold selected on validation F1: "
                    f"{evaluation_report['decision_threshold']:.3f}\n\n"
                )
                summary = pd.DataFrame(
                    [
                        {
                            "count": test_metrics["count"],
                            "accuracy": test_metrics["accuracy"],
                            "balanced accuracy": test_metrics["balanced_accuracy"],
                            "precision": test_metrics["precision"],
                            "recall": test_metrics["recall"],
                            "f1": test_metrics["f1"],
                            "average precision": test_metrics["average_precision"],
                            "TN": test_metrics["true_negative"],
                            "FP": test_metrics["false_positive"],
                            "FN": test_metrics["false_negative"],
                            "TP": test_metrics["true_positive"],
                        }
                    ]
                )
                self.write_column_descriptions(
                    {
                        "count": "Number of test examples.",
                        "accuracy": "Share of all predictions that are correct.",
                        "balanced accuracy": "Mean recall across the positive and negative classes.",
                        "precision": "Share of positive predictions that are correct.",
                        "recall": "Share of positive examples correctly identified.",
                        "f1": "Harmonic mean of precision and recall.",
                        "average precision": "Precision averaged across recall thresholds.",
                        "TN": "True negatives.",
                        "FP": "False positives.",
                        "FN": "False negatives.",
                        "TP": "True positives.",
                    }
                )
                self.report_file.write(summary.to_markdown(index=False, floatfmt=".2f"))
                self.report_file.write("\n\n")
                return

            for evaluation in evaluation_report["evaluations"]:
                if evaluation["label"] == "overall":
                    self.report_file.write(
                        f"Overall accuracy: {evaluation['evaluation']['test']['accuracy']}\n\n"
                    )

    def multiple_boolean_classifier_summary(self, section_conf: ReportSection) -> None:
        if section_conf.models_dir is None:
            return

        model_evaluations: list[Any] = []

        for model in os.listdir(
            os.path.join(
                self.cl_arguments.data_dir,
                "classifiers",
                "models",
                "final",
                "article_models",
                f"{section_conf.models_dir}",
            )
        ):
            model_evaluation = {}

            if model.endswith(".json"):
                with open(
                    os.path.join(
                        self.cl_arguments.data_dir,
                        "classifiers",
                        "models",
                        "final",
                        "article_models",
                        section_conf.models_dir,
                        model,
                    )
                ) as fp:
                    evaluation_report = json.load(fp)
                    model_evaluation["label"] = evaluation_report["objective_label"]

                    model_evaluation["# pos. train"] = evaluation_report["counts"][
                        "positive_train"
                    ]
                    model_evaluation["# neg. train"] = evaluation_report["counts"][
                        "negative_train"
                    ]
                    model_evaluation["# pos. test"] = evaluation_report["counts"][
                        "positive_test"
                    ]
                    model_evaluation["# neg. test"] = evaluation_report["counts"][
                        "negative_test"
                    ]

                    if "metrics" in evaluation_report:
                        test_metrics = evaluation_report["metrics"]["test"]
                        model_evaluation["threshold"] = evaluation_report[
                            "decision_threshold"
                        ]
                        model_evaluation["precision"] = test_metrics["precision"]
                        model_evaluation["recall"] = test_metrics["recall"]
                        model_evaluation["f1"] = test_metrics["f1"]
                        model_evaluation["balanced acc."] = test_metrics[
                            "balanced_accuracy"
                        ]
                        model_evaluation["avg. precision"] = test_metrics[
                            "average_precision"
                        ]

                    acc_pos_train = 0.0
                    acc_neg_train = 0.0
                    acc_pos_train_count = 0
                    acc_neg_train_count = 0
                    acc_pos_test = 0.0
                    acc_neg_test = 0.0
                    acc_pos_test_count = 0
                    acc_neg_test_count = 0
                    for evaluation in evaluation_report["evaluations"]:
                        if evaluation["label"] == "overall":
                            model_evaluation["@ acc. train"] = evaluation["evaluation"][
                                "train"
                            ]["accuracy"]
                            model_evaluation["@ acc. test"] = evaluation["evaluation"][
                                "test"
                            ]["accuracy"]

                        if evaluation["label"] in [
                            "positive_direct",
                            "positive_indirect",
                        ]:
                            acc_pos_train += (
                                evaluation["evaluation"]["train"]["count"]
                                * evaluation["evaluation"]["train"]["accuracy"]
                            )
                            acc_pos_train_count += evaluation["evaluation"]["train"][
                                "count"
                            ]
                            acc_pos_test += (
                                evaluation["evaluation"]["test"]["count"]
                                * evaluation["evaluation"]["test"]["accuracy"]
                            )
                            acc_pos_test_count += evaluation["evaluation"]["test"][
                                "count"
                            ]

                        if evaluation["label"] in [
                            "negative_direct",
                            "negative_indirect",
                        ]:
                            acc_neg_train += (
                                evaluation["evaluation"]["train"]["count"]
                                * evaluation["evaluation"]["train"]["accuracy"]
                            )
                            acc_neg_train_count += evaluation["evaluation"]["train"][
                                "count"
                            ]
                            acc_neg_test += (
                                evaluation["evaluation"]["test"]["count"]
                                * evaluation["evaluation"]["test"]["accuracy"]
                            )
                            acc_neg_test_count += evaluation["evaluation"]["test"][
                                "count"
                            ]

                    model_evaluation["@ acc. pos. train"] = (
                        acc_pos_train / acc_pos_train_count
                    )
                    model_evaluation["@ acc. pos. test"] = (
                        acc_pos_test / acc_pos_test_count
                    )
                    model_evaluation["@ acc. neg. train"] = (
                        acc_neg_train / acc_neg_train_count
                    )
                    model_evaluation["@ acc. neg. test"] = (
                        acc_neg_test / acc_neg_test_count
                    )

                    model_evaluations.append(model_evaluation)
                    fp.close()

        model_evaluations_df: pd.DataFrame = pd.DataFrame(data=model_evaluations)

        model_evaluations_df.sort_values(  # type: ignore
            by="# pos. train", ascending=False, inplace=True
        )

        self.write_column_descriptions(
            {
                "label": "Topic or place predicted by the classifier.",
                "# pos. train": "Positive training examples.",
                "# neg. train": "Negative training examples.",
                "# pos. test": "Positive test examples.",
                "# neg. test": "Negative test examples.",
                "threshold": "Probability cutoff selected on validation F1.",
                "precision": "Share of positive predictions that are correct.",
                "recall": "Share of positive examples correctly identified.",
                "f1": "Harmonic mean of precision and recall.",
                "balanced acc.": "Mean recall across the positive and negative classes.",
                "avg. precision": "Precision averaged across recall thresholds.",
                "@ acc. train": "Overall accuracy on the training set.",
                "@ acc. test": "Overall accuracy on the test set.",
                "@ acc. pos. train": "Accuracy on positive training examples.",
                "@ acc. pos. test": "Accuracy on positive test examples.",
                "@ acc. neg. train": "Accuracy on negative training examples.",
                "@ acc. neg. test": "Accuracy on negative test examples.",
            }
        )
        self.report_file.write(
            model_evaluations_df.to_markdown(index=False, floatfmt=".2f")
        )
        self.report_file.write("\n\n")

    def multiclass_classifier_summary(self, section_conf: ReportSection) -> None:
        if section_conf.model_name is None:
            return

        model_evaluations: list[Any] = []
        with open(
            os.path.join(
                self.cl_arguments.data_dir,
                "classifiers",
                "models",
                "final",
                f"quotes_{section_conf.model_name}_multilabel.json",
            )
        ) as fp:
            evaluation_report = json.load(fp)
            for label, metrics in evaluation_report["metrics_labels"].items():
                model_evaluation: dict[str, Any] = {
                    "label": label,
                    "instances": metrics["tp"] + metrics["fp"] + metrics["fn"],
                    "precision": metrics["precision"]
                    if "precision" in metrics
                    else pd.NA,
                    "recall": metrics["recall"] if "recall" in metrics else pd.NA,
                }
                model_evaluations.append(model_evaluation)

        model_evaluations_df = pd.DataFrame(data=model_evaluations)
        model_evaluations_df.sort_values(by="instances", ascending=False, inplace=True)  # type: ignore

        # Handle pd.NA values for float formatting
        model_evaluations_df_display = model_evaluations_df.copy()
        model_evaluations_df_display["precision"] = model_evaluations_df_display[
            "precision"
        ].fillna(0.0)  # type: ignore
        model_evaluations_df_display["recall"] = model_evaluations_df_display[
            "recall"
        ].fillna(0.0)  # type: ignore

        model_evaluations_df_display["f1-score"] = (
            2
            * (
                model_evaluations_df_display["precision"]
                * model_evaluations_df_display["recall"]
            )
            / (
                model_evaluations_df_display["precision"]
                + model_evaluations_df_display["recall"]
            )
        )

        self.write_column_descriptions(
            {
                "label": "Type, tone, topic, or place predicted by the classifier.",
                "instances": "Test examples associated with the label.",
                "precision": "Share of predictions for the label that are correct.",
                "recall": "Share of examples with the label correctly identified.",
                "f1-score": "Harmonic mean of precision and recall.",
            }
        )
        self.report_file.write(
            model_evaluations_df_display.to_markdown(index=False, floatfmt=".2f")
        )
        self.report_file.write("\n\n")


def parse_command_line_arguments() -> CommandLineArguments:
    parser = argparse.ArgumentParser(
        description="Summary generation for multiple classifiers"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing training data and output models",
    )
    args = parser.parse_args()

    arguments = CommandLineArguments(data_dir=args.data_dir)

    # Initialize data directory
    if not os.path.isdir(arguments.data_dir):
        # Argument is not a directory
        logging.error(f"--data_dir argument is not a directory: {arguments.data_dir}")
        exit(1)

    return arguments


if __name__ == "__main__":
    # Read command line arguments
    args = parse_command_line_arguments()

    # The structure of the report
    # FIXME: Read from command line argument
    report_structure = EvaluationReport(
        path="reports/urbanixm-classifiers.md",
        sections=[
            ReportSection(
                type="single-boolean",
                title="Article Urbanism Relevance",
                model_name="on_topic",
            ),
            ReportSection(
                type="single-boolean",
                title="Article Quotability",
                model_name="quotable",
            ),
            ReportSection(
                type="multiple-boolean", title="Article Topics", models_dir="topics"
            ),
            ReportSection(
                type="multiple-boolean", title="Article Places", models_dir="places"
            ),
            ReportSection(
                type="multilabel", title="Quote types", model_name="quote_types"
            ),
            ReportSection(
                type="multilabel", title="Quote tones", model_name="quote_tones"
            ),
            ReportSection(
                type="multilabel", title="Quote topics", model_name="quote_topics"
            ),
            ReportSection(
                type="multilabel", title="Quote places", model_name="quote_places"
            ),
        ],
    )

    summarizer = EvaluationSummarizer(args=args)
    summarizer.generate_summary(report_conf=report_structure)
