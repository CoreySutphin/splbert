import os
import sys
import json
import torch
import numpy as np
from spacy.gold import biluo_tags_from_offsets
from transformers import BertTokenizerFast, AlbertTokenizer


def encode_labels_for_subwords(
    labels, encodings,
):
    """Set the first subword of each word as the tag, and each subsequent subword as -100.

    BERT uses WordPiece tokenization, so a single token could be split up into multiple sub-word tokens.
    This can cause issues since our tags are associated with a signle word token, so we could end up with a mismatch
    between our tokens and our tags.

    TODO: How to handle this for dis-contiguous entities? Specifically how do you merge tokens back if subwords belong to different
    tokens are nested next to each other?
    """
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        print(arr_offset)
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


class TACDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        return

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

    @staticmethod
    def from_spacy_encodings(
        json_directory, tokenizer, save_directory=None, max_sequence_length=512,
    ):
        """
            Given a list of tuples of document with span level annotations, saves bert input and labels onto disk.
            This method is designed as a pre-processing step to be utilized with a pytorch Dataset and Dataloader.
            :param data:  a list of tuples relating a document to its set of annotations.
            :param tokenizer: the transformers tokenizer to utilize.
            :return the location the dataset was saved

        """

        # Load in all the SpaCy docs + token annotations from the JSON files
        documents = []
        unique_tags = set()
        token_sequences = []
        label_sequences = []
        misaligned_tags = 0
        for filename in os.listdir(json_directory):
            with open(os.path.join(json_directory, filename), "r") as f:
                document = json.loads(f.read())
                documents.append(document)

                # Collect all tags from the dataset in a set
                for tag in document["tags"]:
                    if tag == "-":
                        misaligned_tags += 1
                    unique_tags.add(tag)

                label_sequences.append(document["tags"])
                token_sequences.append([x for x in document["tokens"]])

        tag2id = {tag: id for id, tag in enumerate(unique_tags)}
        id2tag = {id: tag for tag, id in tag2id.items()}
        print(tag2id)
        print(id2tag)
        print(f"Misaligned Tags: {misaligned_tags}")
        # print(label_sequences)

        # Encode the label sequences from strings to IDs
        encoded_labels = []

        # Tokenize our tokens for sub-words using the supplied tokenizer
        token_sequence_encodings = tokenizer(
            token_sequences,
            is_pretokenized=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )
        # print(token_encodings)

        l = [[tag2id[tag] for tag in doc] for doc in label_sequences]
        encoded_train_labels = encode_labels_for_subwords(l, token_sequence_encodings)
        print(encoded_train_labels)

        return

        # biluo_ordered_labels = sorted(
        #     [
        #         f"{prefix}-{entity_label}"
        #         for prefix in ["B", "I", "L", "U"]
        #         for entity_label in entity_labels
        #         if entity_label != "O"
        #     ]
        #     + ["O", "BERT_TOKEN"]
        # )
        # tags_from_annotations = sorted(list(tags_from_annotations) + ["BERT_TOKEN"])
        #
        # # convert each string label to a unique id with respect to the biluo_labels of the tokenization
        # encoded_label_sequences = [
        #     [biluo_ordered_labels.index(label) for label in seq]
        #     for seq in label_sequences
        # ]
        #
        # class_counts = [0] * len(biluo_ordered_labels)
        #
        # for seq in encoded_label_sequences:
        #     for id in seq:
        #         class_counts[id] += 1
        #
        # class_counts = torch.FloatTensor(class_counts)
        # loss_weights = torch.abs(
        #     1
        #     - (class_counts / len([x for x in seq for seq in encoded_label_sequences]))
        # )
        # # Assert that all labels appear in the annotations. This could occur if annotation processing could not align
        # # all annotations into the defined spacy tokenization.
        # if biluo_ordered_labels != tags_from_annotations:
        #     warnings.warn(
        #         "Processed dataset does not contain instances from all labels when converted to BILOU scheme."
        #     )
        #
        # # Now generate bert input tensors
        # (
        #     all_bert_sequence_alignments,
        #     all_bert_subword_sequences,
        #     all_bert_label_sequences,
        #     original_tokenization_labels,
        # ) = ([], [], [], [])
        #
        # for sequence, labels in zip(token_sequences, encoded_label_sequences):
        #
        #     # alignment from the bert tokenization to spaCy tokenization
        #     assert len(sequence) == len(labels)
        #
        #     # maps each original token to it's subwords
        #     token_idx_to_subwords = []
        #     for token in sequence:
        #         token_idx_to_subwords.append(
        #             [subword for subword in tokenizer.tokenize(str(token))]
        #         )
        #
        #     bert_subwords = ["[CLS]", "[SEP]"]
        #     bert_subword_labels = [
        #         biluo_ordered_labels.index("BERT_TOKEN"),
        #         biluo_ordered_labels.index("BERT_TOKEN"),
        #     ]
        #     bert_subword_to_original_tokenization_alignment = [-1, -1]
        #     original_tokens_processed = []
        #
        #     idx = 0
        #     chunk_start = 0
        #     while idx < len(sequence):
        #
        #         start_next_buffer = False
        #         token_in_buffer_size = (
        #             len(bert_subwords) + len(token_idx_to_subwords[idx])
        #             <= max_sequence_length
        #         )
        #
        #         if token_in_buffer_size:
        #             # build a sequence
        #             bert_subwords[-1:-1] = [
        #                 subword for subword in token_idx_to_subwords[idx]
        #             ]
        #             bert_subword_labels[-1:-1] = [
        #                 labels[idx] for _ in token_idx_to_subwords[idx]
        #             ]
        #             bert_subword_to_original_tokenization_alignment[-1:-1] = [
        #                 idx - chunk_start for _ in token_idx_to_subwords[idx]
        #             ]
        #             original_tokens_processed.append(idx)
        #             idx += 1
        #
        #         # Insure we aren't splitting on a label by greedily splitting on 'O' labels once the buffer gets very full (>500 subwords)
        #         if len(bert_subwords) > 500 and labels[
        #             idx - 1
        #         ] == biluo_ordered_labels.index("O"):
        #             start_next_buffer = True
        #
        #         if not token_in_buffer_size or start_next_buffer:
        #             all_bert_subword_sequences.append(bert_subwords)
        #             all_bert_label_sequences.append(bert_subword_labels)
        #             all_bert_sequence_alignments.append(
        #                 bert_subword_to_original_tokenization_alignment
        #             )
        #
        #             original_tokenization_labels.append(
        #                 [labels[i] for i in original_tokens_processed]
        #             )
        #
        #             # reset sequence builders
        #             bert_subwords = ["[CLS]", "[SEP]"]
        #             bert_subword_labels = [
        #                 biluo_ordered_labels.index("BERT_TOKEN"),
        #                 biluo_ordered_labels.index("BERT_TOKEN"),
        #             ]
        #             bert_subword_to_original_tokenization_alignment = [-1, -1]
        #             original_tokens_processed = []
        #             chunk_start = idx
        #
        #     if bert_subwords != ["[CLS]", "[SEP]"]:
        #         # Add the remaining
        #         all_bert_subword_sequences.append(bert_subwords)
        #         all_bert_label_sequences.append(bert_subword_labels)
        #         all_bert_sequence_alignments.append(
        #             bert_subword_to_original_tokenization_alignment
        #         )
        #         original_tokenization_labels.append(
        #             [labels[i] for i in original_tokens_processed]
        #         )
        #
        # for seq in original_tokenization_labels:
        #     for label in seq:
        #         assert label != -1
        #
        # max_num_spacy_labels = max([len(seq) for seq in original_tokenization_labels])
        #
        # bert_input_ids = torch.zeros(
        #     size=(len(all_bert_subword_sequences), max_sequence_length),
        #     dtype=torch.long,
        # )
        # bert_attention_masks = torch.zeros_like(bert_input_ids)
        # bert_sequence_lengths = torch.zeros(size=(len(all_bert_subword_sequences), 1))
        #
        # bert_labels = torch.zeros_like(bert_input_ids)
        # bert_alignment = torch.zeros_like(bert_input_ids)
        # gold_original_token_labels = torch.zeros(
        #     size=(len(all_bert_subword_sequences), max_num_spacy_labels),
        #     dtype=torch.long,
        # )
        #
        # for (
        #     idx,
        #     (
        #         bert_subword_sequence,
        #         bert_label_sequence,
        #         alignment,
        #         original_tokenization_label,
        #     ),
        # ) in enumerate(
        #     zip(
        #         all_bert_subword_sequences,
        #         all_bert_label_sequences,
        #         all_bert_sequence_alignments,
        #         original_tokenization_labels,
        #     )
        # ):
        #     if len(bert_subword_sequence) > 512:
        #         raise BaseException(
        #             "Error sequence at index %i as it is to long (%i tokens)"
        #             % (idx, len(bert_subword_sequence))
        #         )
        #     input_ids = tokenizer.convert_tokens_to_ids(bert_subword_sequence)
        #     attention_masks = [1] * len(input_ids)
        #
        #     while (
        #         len(input_ids) < max_sequence_length
        #     ):  # pad bert aligned input until max length
        #         input_ids.append(0)
        #         attention_masks.append(0)
        #         bert_label_sequence.append(0)
        #         alignment.append(-1)
        #     while (
        #         len(original_tokenization_label) < max_num_spacy_labels
        #     ):  # pad spacy aligned input with -1
        #         original_tokenization_label.append(-1)
        #
        #     bert_input_ids[idx] = torch.tensor(input_ids, dtype=torch.long)
        #     bert_attention_masks[idx] = torch.tensor(attention_masks, dtype=torch.long)
        #     bert_alignment[idx] = torch.tensor(alignment, dtype=torch.long)
        #     bert_sequence_lengths[idx] = torch.tensor(
        #         sum([1 for x in input_ids if x != 0]), dtype=torch.long
        #     )
        #     gold_original_token_labels[idx] = torch.tensor(
        #         original_tokenization_label, dtype=torch.long
        #     )
        #     bert_labels[idx] = torch.tensor(bert_label_sequence, dtype=torch.long)
        #
        #     for i in range(1, len(bert_labels[idx]) - 1):
        #         # print()
        #         # print(f"Bert Labels | {i} | {bert_labels[idx][i]}")
        #         # print(f"Correct Original Labels | {i} | {gold_original_token_labels[idx][bert_alignment[idx][i]]}")
        #         # print(f"Bert Labels: {bert_labels[idx]}")
        #         # print(f"Spacy Labels: {gold_original_token_labels[idx]}")
        #         # print(f"Bert Alignment: {bert_alignment[idx]}")
        #         try:
        #             assert (
        #                 bert_labels[idx][i]
        #                 == gold_original_token_labels[idx][bert_alignment[idx][i]]
        #             )
        #         except BaseException:
        #             pass
        # if save_directory:
        #     torch.save(
        #         bert_input_ids, os.path.join(save_directory, f"bert_input.pt")
        #     )  # bert input ids
        #     torch.save(
        #         bert_attention_masks,
        #         os.path.join(save_directory, f"bert_attention_mask.pt"),
        #     )  # bert attention masks
        #     torch.save(
        #         bert_sequence_lengths,
        #         os.path.join(save_directory, f"bert_sequence_length.pt"),
        #     )  # length of actual bert sequence
        #     torch.save(
        #         bert_labels, os.path.join(save_directory, f"bert_labels.pt")
        #     )  # correct labels relative to bert tokenization
        #     torch.save(
        #         gold_original_token_labels,
        #         os.path.join(save_directory, f"spacy_labels.pt"),
        #     )  # correct labels relative to spacy tokenization
        #     torch.save(
        #         bert_alignment,
        #         os.path.join(save_directory, f"subword_to_spacy_alignment.pt"),
        #     )  # alignment between bert and spacy sequences
        #     torch.save(
        #         biluo_ordered_labels, os.path.join(save_directory, "entity_names.pl")
        #     )  # entity labels
        #     torch.save(
        #         loss_weights, os.path.join(save_directory, "loss_weights.pt")
        #     )  # global entity class counts
        #
        # return (
        #     (bert_input_ids, None, bert_attention_masks),
        #     bert_sequence_lengths,
        #     bert_labels,
        #     original_tokenization_labels,
        #     bert_alignment,
        #     biluo_ordered_labels,
        #     loss_weights,
        # )


directory = sys.argv[1]
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
dataset = TACDataset.from_spacy_encodings(directory, tokenizer)
