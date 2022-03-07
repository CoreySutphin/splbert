import os
import torch
from torch.nn import Linear, Module, CrossEntropyLoss

from .base_head import BaseHead


class RelationshipExtractionHead(BaseHead):
    """TODO: IMPLEMENT ME"""

    def __init__(self, labels, hidden_size=1536):
        """"""
        super(RelationshipExtractionHead, self).__init__(
            name="NerHead",
            labels=labels,
            hidden_size=hidden_size,
        )
        self.classifier = Linear(hidden_size, len(self.config.labels))
