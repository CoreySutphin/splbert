import os
import torch
from torch.nn import Linear, Module, CrossEntropyLoss

from .base_head import BaseHead


class EntityRecognitionHead(BaseHead):
    """
    TODO: This was copied over from renerbert, it needs to be tested and cleaned
    """

    def __init__(self, labels=None, hidden_size=768):
        super(EntityRecognitionHead, self).__init__(
            name="NerHead",
            labels=labels,
            hidden_size=hidden_size,
        )
        self.classifier = Linear(hidden_size, len(self.config.labels))

    def forward(
        self,
        outputs,
        attention_mask=None,
        labels=None,
        loss_weights=None,
    ):
        """
        :param hidden_states:
        :param attention_mask:
        :param labels:
        :param loss_weights:
        :return:
        """

        hidden_states = outputs["last_hidden_state"]
        logits = self.classifier(hidden_states)
        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=loss_weights)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, len(self.config.labels))[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, len(self.config.labels)), labels.view(-1)
                )

        return (logits, loss)

    def save(self, head_directory: str):
        config_filepath = os.path.join(
            head_directory,
            f"{self.config.head_name}_{self.config.head_task}.json".replace(" ", "_"),
        )
        weight_filepath = os.path.join(
            head_directory,
            f"{self.config.head_name}_{self.config.head_task}.pt".replace(" ", "_"),
        )

        self.config.to_json_file(config_filepath)
        torch.save(self.state_dict(), weight_filepath)
