"""
Class to represent the tokens, Mention labels, and Relationship labels of the TAC 2019 DDI Dataset.

To create tensors used to instantiate the Dataset, run this file as a python script as follows:
python tac_dataset.py input_directory output_directory
"""
import os
import sys
import json
import torch
import numpy as np
from transformers import BertTokenizerFast, AlbertTokenizer


class TACDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.directory = directory

        # Load in data pre-processed using BERT
        self.input_ids = torch.load(os.path.join(self.directory, "input_ids.pt"))
        self.attention_masks = torch.load(
            os.path.join(self.directory, "attention_masks.pt")
        )
        self.offset_mappings = torch.load(
            os.path.join(self.directory, "offset_mappings.pt")
        )
        self.ner_labels = torch.load(os.path.join(self.directory, "ner_labels.pt"))
        # self.re_labels = torch.load(os.path.join(self.directory, "re_labels.pt"))

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_masks": self.attention_masks[idx],
            "offset_mappings": self.offset_mappings[idx],
            "labels": torch.tensor(self.labels[idx]),
            # "re_labels": self.re_labels[idx],
        }

    def __len__(self):
        return len(self.input_ids)

    @classmethod
    def tokenize_and_align_labels(cls, token_encodings, label_seqs):
        """Given subword-encoded tokens, align labels onto subwords.
        The first subword of a given token is overlaid with the label for that token, everything
        else is encoded as -100.

        Args:
            token_encodings (transformers.BatchEncoding) - BERT-encoded subword tokens
            label_seqs (list of lists) - List of label sequences, aligned with original tokenization

        Returns:
            (list of lists) - List of label sequences aligned with subword tokenization

        """
        label_encodings = []
        for i, label_seq in enumerate(label_seqs):
            word_ids = token_encodings[i].ids
            offsets = token_encodings[i].offsets
            label_index = 0
            encoded_label_ids = []
            for i, word_idx in enumerate(word_ids):
                # These word ids represent special token([CLS], [SEP], or [PAD])
                if word_idx in [0, 101, 102]:
                    encoded_label_ids.append(-100)
                elif offsets[i][0] != 0:
                    # We currently set only the first subword to the class label. Everything
                    # else is encoded as -100.
                    encoded_label_ids.append(-100)
                else:
                    encoded_label_ids.append(label_seq[label_index])
                    label_index += 1
            label_encodings.append(encoded_label_ids)

        return label_encodings

    @classmethod
    def create_from_spacy_encodings(
        cls,
        json_directory,
        tokenizer,
        save_directory=None,
        max_sequence_length=512,
    ):
        """
        Given a list of tuples of document with span level annotations, saves bert input and labels onto disk.
        This method is designed as a pre-processing step to be utilized with a pytorch Dataset and Dataloader.
        :param data:  a list of tuples relating a document to its set of annotations.
        :param tokenizer: the transformers tokenizer to utilize.
        :return the location the dataset was saved

        1. Tokenize our labels into an id2tag and tag2id dict
        2. Tokenize each sentence, keeping in mind padding and special characters([CLS], [PAD], [SEP])
        3. Align tokenized labels over bert-tokenized subwords. We will set the first subword as the label and the rest
           will be -100(ignored during training).
        4.

        """
        unique_labels = set()
        token_sequences = []
        label_sequences = []
        misaligned_tags = 0

        # For each document, collect the unique labels, as well as the token and label sequences in the document.
        for filename in os.listdir(json_directory):
            with open(os.path.join(json_directory, filename), "r") as f:
                document = json.loads(f.read())

                # Collect all tags from the dataset in a set
                for label_seq in document["labels"]:
                    for label in label_seq:
                        unique_labels.add(label)

                token_sequences += document["tokens"]
                label_sequences += document["labels"]

        # Generate a mapping of labels to int IDs.
        label2id = {label: id for id, label in enumerate(sorted(unique_labels))}
        id2label = {id: label for label, id in label2id.items()}
        print(f"Label 2 ID: {label2id}")
        print(f"ID 2 Label: {id2label}")

        # Tokenize our tokens for sub-words using the supplied tokenizer
        token_encodings = tokenizer(
            token_sequences,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )

        print(token_encodings[0].tokens)
        print(tokenizer.decode(token_encodings[0].ids))

        label_encodings = [
            [label2id[label] for label in label_seq] for label_seq in label_sequences
        ]
        print(label_sequences[0])
        print(label_encodings[0])

        subword_label_encodings = cls.tokenize_and_align_labels(
            token_encodings, label_encodings
        )
        print(subword_label_encodings[0])

        bert_input_ids = token_encodings["input_ids"]
        bert_attention_masks = token_encodings["attention_mask"]
        bert_offset_mappings = token_encodings["offset_mapping"]

        # Save the output to disk
        if save_directory:
            torch.save(bert_input_ids, os.path.join(save_directory, "input_ids.pt"))
            torch.save(
                bert_attention_masks, os.path.join(save_directory, "attention_masks.pt")
            )
            torch.save(
                bert_offset_mappings, os.path.join(save_directory, "offset_mappings.pt")
            )
            torch.save(
                subword_label_encodings, os.path.join(save_directory, "ner_labels.pt")
            )
            # torch.save(re_labels, os.path.join(save_directory, "re_labels.pt"))

        return (
            bert_input_ids,
            bert_attention_masks,
            bert_offset_mappings,
            subword_label_encodings,
        )


if __name__ == "__main__":
    input_directory = sys.argv[1]
    if len(sys.argv) > 2:
        output_directory = sys.argv[2]
    else:
        output_directory = None
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    dataset = TACDataset.create_from_spacy_encodings(
        input_directory, tokenizer, save_directory=output_directory
    )
