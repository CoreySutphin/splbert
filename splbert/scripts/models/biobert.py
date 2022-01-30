"""
    Fine-tune the BioBert Transformers model on a given training set, and then perform predictions on a given test set.

    To Use:
    python run_biobert.py training_directory test_directory output_directory device(-1 for CPU, otherwise put GPU identified)
"""

import os
import sys
import json
import logging

import numpy as np

from transformers import BertTokenizer, AlbertTokenizer
from medacy.pipeline_components.learners.bert_learner import BertLearner

assert len(sys.argv) > 1, "Please pass in the directory of documents"

test_directory = sys.argv[1]
output_directory = sys.argv[2]
model_path = sys.argv[3]

device = int(sys.argv[4]) if len(sys.argv) > 4 else -1

# Initialize the BioBERT learner
learner = BertLearner(
    cuda_device=device, pretrained_model="monologg/biobert_v1.1_pubmed"
)

# Load the pre-trained weights
learner.load(model_path)

# Load in each test files and do predictions
for filename in os.listdir(test_directory):
    logging.info(f"Predicting file {filename}")
    test_sequences = []
    with open(os.path.join(test_directory, filename), "r") as f:
        document = json.loads(f.read())

        # Collect token sequences from the document
        for token_sequence in document["tokens"]:
            # Truncate the sequence to the first 511 tokens - because BERT cannot handle more than 512
            token_sequence = token_sequence[:511]
            test_sequences.append(token_sequence)

    # Perform prediction
    predictions = learner.predict(test_sequences)

    # Create Mention instances

    with open(os.path.join(output_directory, filename), "w+") as out:
        document.pop("labels", None)
        output = {"labels": predictions, **document}
        out.write(json.dumps(output))
