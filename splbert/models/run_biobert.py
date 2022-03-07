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

from utils import cross_validate

# Set logger level to INFO
logging.getLogger().setLevel(logging.INFO)

assert len(sys.argv) > 1, "Please pass in the directory of documents"

training_directory = sys.argv[1]
test_directory = sys.argv[2]
output_directory = sys.argv[3]

device = int(sys.argv[4]) if len(sys.argv) > 4 else -1


# Initialize the BioBERT learner
learner = BertLearner(
    cuda_device=device, pretrained_model="monologg/biobert_v1.1_pubmed"
)

# Load in training sequences of tokens and labels
unique_labels = set()
token_sequences = []
label_sequences = []
correct_labels = 0
misaligned_labels = 0

for filename in os.listdir(training_directory):
    with open(os.path.join(training_directory, filename), "r") as f:
        document = json.loads(f.read())

        # Collect label sequences from the document
        for label_sequence in document["labels"]:
            # Truncate the sequence to the first 511 tokens - because BERT cannot handle more than 512
            label_sequence = label_sequence[:511]
            label_sequences.append(label_sequence)
            for label in label_sequence:
                if label == "-":
                    misaligned_labels += 1
                    continue
                unique_labels.add(label)
                if label != "-" and label != "O":
                    correct_labels += 1

        # Collect token sequences from the document
        for token_sequence in document["tokens"]:
            # Truncate the sequence to the first 511 tokens - because BERT cannot handle more than 512
            token_sequence = token_sequence[:511]
            token_sequences.append(token_sequence)

print(f"Number of mis-aligned labels: {misaligned_labels}\n")
print(f"Number of correctly labeled entities: {correct_labels}\n")
print(f"Number of sequences: {len(label_sequences)}\n")

# Train the model
# print("Label Sequences", label_sequences)
# print("Token Sequences", token_sequences)
learner.fit(token_sequences, label_sequences)

# Store the trained model to disk
learner.save("./biobert_standard.out")

exit()

# Load in each test file and do predictions
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
        output = {"labels": predictions, **document}
        out.write(json.dumps(output))
