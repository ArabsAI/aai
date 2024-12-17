import torch
import torch.nn as nn

from aai.metrics import Metric


class CrossEntropyLoss(Metric):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
