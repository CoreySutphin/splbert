import transformers
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, CONFIG_NAME, BertModel, WEIGHTS_NAME
from torch.utils.data import DataLoader

from ..heads import EntityRecognitionHead, RelationshipExtractionHead


class JointTaskModel:
    def init(
        self,
        id_to_label_map,
        device=-1,
    ):
        """Model to jointly train an NER head and RE head both using the same BERT encoder.

        Args:
            id_to_label_map (dict) - Mapping of numerical IDs to entity labels.
            device (int) - Device to use, defaults to CPU.
        """
        self.device = device
        self.optimizer = None  # TODO: Set this optimizer

        # TODO: Initialize the base BERT encoder
        self.bert = BertModel.from_pretrained("bert-based-cased")

        # TODO: Create this head
        self.ner_head = EntityRecognitionHead(labels=id_to_label_map.keys())

        # TODO: Create this head
        self.re_head = RelationshipExtractionHead()

    def fit(self, train_loader, epochs=3):
        """Fit the joint model on a given dataset.
        Both the BERT encoder and individual task head weights are learned.

        Args:
            train_loader (DataLoader) - PyTorch DataLoader to iterate through our training dataset in batches.
            epochs (int) - Number of epochs to train for, defaults to 3 as recommended by the original BERT paper.
        """

        # Configure our BERT encoder and heads for training
        self.bert.train()
        self.bert.to(device=self.device)
        self.ner_head.train()
        self.ner_head.to(device=self.device)
        self.re_head.train()
        self.re_head.to(device=self.device)

        # Training loop
        for i in range(epochs):
            for batch_idx, batch in enumerate(train_loader):

                # Load BERT inputs on to the device
                input_ids = batch["input_ids"].to(device=self.device)
                attention_masks = batch["attention_masks"].to(device=self.device)
                ner_labels = batch["ner_labels"].to(device=self.device)
                re_labels = batch["re_labels"].to(device=self.device)

                # Get the BERT encodings
                bert_outputs = self.bert(
                    input_ids=input_ids,
                    attention_masks=attention_masks,
                )

                # Get the outputs from NER
                ner_outputs, ner_loss = self.ner_head(bert_outputs, ner_labels)

                # TODO: Pair together predicted labels that could have a relationship.
                #       Our specific task(TAC 2019 DDI) has restrictions on what entity paris can have relationships between each other.

                # Gather entity pairs, concat their embeddings together, and feed through the RE head
                re_inputs = None
                re_outputs, re_loss = self.re_head(re_inputs, re_labels)

                # TODO: Right now we are just using additive loss between the NER and RE head, allow for other loss options.
                # TODO: Implement additive loss, https://discuss.pytorch.org/t/combine-losses-and-weight-those/108700
                loss = ner_loss + re_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        return

    def predict(self):
        """TODO: IMPLEMENT ME"""
        pass
