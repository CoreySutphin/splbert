import os
import json
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, Linear, Dropout, Module

"""
TODO: This was copied over from renerbert, need to test and clean this up.
"""


class HeadConfig(dict):
    def __init__(self, *args, **kwargs):
        super(HeadConfig, self).__init__(*args, **kwargs)
        self.__dict__ = self

        assert hasattr(self, "name"), "Head Config requires a name."

    def to_disk(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(json.dumps(self))

    @classmethod
    def from_disk(cls, filepath: str):
        with open(filepath, "r") as f:
            obj = json.loads(f.read())
            return cls(**obj)

    def __str__(self):
        return f"{self.name}".replace(" ", "_")


class BaseHead(Module):
    def __init__(self, *args, **kwargs):
        super(BaseHead, self).__init__()
        self.config = HeadConfig(*args, **kwargs)

    @classmethod
    def from_disk(
        cls,
        config_path: str,
        weights_path: str,
    ):
        # Load the config for the head
        config = HeadConfig.from_disk(config_path)
        head = cls(**config.__dict__)

        # Load the weights of the head
        if torch.cuda.is_available():
            head.load_state_dict(torch.load(weights_path))
        else:
            head.load_state_dict(
                torch.load(weights_path, map_location=torch.device("cpu"))
            )

        return head

    def to_disk(
        self,
        config_path: str,
        weights_path: str,
    ):
        self.config.to_disk(config_path)
        torch.save(self.state_dict(), weights_path)

    def __str__(self):
        return self.config.__str__()

    def __repr__(self):
        return self.config.__repr__()
