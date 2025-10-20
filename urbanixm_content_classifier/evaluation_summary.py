import argparse
from dataclasses import dataclass
from io import TextIOWrapper
import json
import logging
import os
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


    def generate_summary(self, report_conf: EvaluationReport) -> None:

        self.report_file = open(report_conf.path, 'w')

        for section in report_conf.sections:

            self.report_file.write(f'## {section.title}\n\n')

            if section.type == 'single-boolean':  # FIXME: get from enum
                self.single_boolean_classifier_summary(
                    section_conf=section
                )
            elif section.type == 'multiple-boolean': # FIXME: get from enum
                self.multiple_boolean_classifier_summary(
                    section_conf=section
                )
            elif section.type == 'multilabel': # FIXME: get from enum
                self.multiclass_classifier_summary(
                    section_conf=section
                )
        self.report_file.close()

    def single_boolean_classifier_summary(self, section_conf: ReportSection) -> None:
        
        if section_conf.model_name is not None:

            with(open(os.path.join(self.cl_arguments.data_dir, 
                                   'classifiers', 
                                   'models', 
                                   'final', 
                                   'article_models',
                                   section_conf.model_name,
                                   f"{section_conf.model_name}.json"))) as fp:
                evaluation_report = json.load(fp)
                fp.close()
        
            for evaluation in evaluation_report['evaluations']:
                if evaluation['label'] == 'overall':
                    self.report_file.write(f'Overall accuracy: {evaluation['evaluation']['test']['accuracy']}\n\n')

    def multiple_boolean_classifier_summary(self, section_conf: ReportSection) -> None:

        if section_conf.models_dir is None:
            return

        model_evaluations = []
        for model in os.listdir(os.path.join(self.cl_arguments.data_dir, 
                                             'classifiers', 
                                             'models', 
                                             'final', 
                                             'article_models',
                                             f'{section_conf.models_dir}')):
            model_evaluation = {}

            if model.endswith('.json'):
                with(open(os.path.join(self.cl_arguments.data_dir, 
                                   'classifiers', 
                                   'models', 
                                   'final', 
                                   'article_models',
                                   section_conf.models_dir,
                                   model))) as fp:
                    evaluation_report = json.load(fp)
                    model_evaluation['label'] = evaluation_report['objective_label']

                    model_evaluation['# pos. train'] = evaluation_report['counts']['positive_train']
                    model_evaluation['# neg. train'] = evaluation_report['counts']['negative_train']
                    model_evaluation['# pos. test'] = evaluation_report['counts']['positive_test']
                    model_evaluation['# neg. test'] = evaluation_report['counts']['negative_test']

                    for evaluation in evaluation_report['evaluations']:
                        if evaluation['label'] == 'overall':
                            model_evaluation['@ acc. train'] = evaluation['evaluation']['train']['accuracy']
                            model_evaluation['@ acc. test'] = evaluation['evaluation']['test']['accuracy']

                    model_evaluations.append(model_evaluation)
                    fp.close()

        model_evaluations_df = pd.DataFrame(data=model_evaluations)

        model_evaluations_df.sort_values(by='# pos. train', ascending=False, inplace=True)

        self.report_file.write(model_evaluations_df.to_markdown(floatfmt=".2f"))
        self.report_file.write("\n\n")

    def multiclass_classifier_summary(self, section_conf: ReportSection) -> None:

        if section_conf.model_name is None:
            return

        model_evaluations = []
        with(open(os.path.join(self.cl_arguments.data_dir, 
                                   'classifiers', 
                                   'models', 
                                   'final', 
                                   f'quotes_{section_conf.model_name}_multilabel.json'))) as fp:
            evaluation_report = json.load(fp)
            for label, metrics in evaluation_report['metrics_labels'].items():
                model_evaluation = {
                    "label": label,
                    "instances": metrics['tp'] + metrics['fp'] + metrics['fn'],
                    "precision": metrics['precision'] if 'precision' in metrics else pd.NA,
                    "recall": metrics['recall'] if 'recall' in metrics else pd.NA, 
                }
                model_evaluations.append(model_evaluation)
        
        model_evaluations_df = pd.DataFrame(data=model_evaluations)
        model_evaluations_df.sort_values(by="instances", ascending=False, inplace=True)

        # Handle pd.NA values for float formatting
        model_evaluations_df_display = model_evaluations_df.copy()
        model_evaluations_df_display['precision'] = model_evaluations_df_display['precision'].fillna(0.0)
        model_evaluations_df_display['recall'] = model_evaluations_df_display['recall'].fillna(0.0)

        model_evaluations_df_display['f1-score'] = \
                    2 * \
                    (model_evaluations_df_display['precision'] * model_evaluations_df_display['recall']) \
                    / (model_evaluations_df_display['precision'] + model_evaluations_df_display['recall'])
                    
        self.report_file.write(model_evaluations_df_display.to_markdown(floatfmt=".2f"))
        self.report_file.write("\n\n")


def parse_command_line_arguments() -> CommandLineArguments:
    parser = argparse.ArgumentParser(
    description="Summary generation for multiple classifiers")
    parser.add_argument("--data_dir", 
                        type=str, 
                        required=True, 
                        help="Path to the directory containing training data and output models")
    args = parser.parse_args()
    
    arguments = CommandLineArguments(
        data_dir=args.data_dir
    ) 

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
                model_name="on_topic"
            ),
            ReportSection(
                type="single-boolean",
                title="Article Quotability",
                model_name="quotable"
            ),
            ReportSection(
                type="multiple-boolean",
                title="Article Topics",
                models_dir='topics'
            ),
            ReportSection(
                type="multiple-boolean",
                title="Article Places",
                models_dir='places'
            ),
            ReportSection(
                type="multilabel",
                title="Quote types",
                model_name='quote_types'
            ),
            ReportSection(
                type="multilabel",
                title="Quote tones",
                model_name='quote_tones'
            ),
            ReportSection(
                type="multilabel",
                title="Quote topics",
                model_name='quote_topics'
            ),
            ReportSection(
                type="multilabel",
                title="Quote places",
                model_name='quote_places'
            ),
        ]
    )

    summarizer = EvaluationSummarizer(args=args)
    summarizer.generate_summary(report_conf=report_structure)
