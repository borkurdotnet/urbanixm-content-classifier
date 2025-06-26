import json
import os
import sys
from urbanixm_content_classifier.quote_classifier import QuoteClassificationTrainer, ObjectiveType

def test_train():
    """
    Test the training of a classifier

    Warning: This test can take a while (~30s)!
    """
    
    # Clean existing model
    import shutil
    model_dirs = [
        'tests/data/classifiers/models/final/quotes_quote_types_multilabel',
        'tests/data/classifiers/models/interim/quotes_quote_types_multilabel'
    ]
    model_files = [
        'tests/data/classifiers/models/final/quotes_quote_types_multilabel.json'
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
    
    for model_file in model_files:
        if os.path.exists(model_file):
            os.remove(model_file)

    # Mock command line arguments
    sys.argv = ['test_script', '--data_dir', 'tests/data', '--objective_type', 'quote_types']
    
    trainer = QuoteClassificationTrainer()
    trainer.parse_arguments_training()
    trainer.train()

    with open(model_files[0], 'r') as fp:
        model_metadata = json.load(fp)

        assert len(model_metadata['label_names']) == 5
        assert "impact" in model_metadata['label_names']
        assert "informative" in model_metadata['label_names']
        assert "infrastructure" in model_metadata['label_names']
        assert "off-topic" in model_metadata['label_names']
        assert "policy" in model_metadata['label_names']

