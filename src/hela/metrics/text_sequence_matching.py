"""Implementation of accuracy metric based on entire text sequence matching."""

import torch
from torch import Tensor
from torchmetrics import Metric


class SequenceMatchingMetric(Metric):
    """Sequence Matching Metric."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.correct += sum(torch.all(preds == target, dim=1))
        self.total += preds.shape[0]

    def compute(self) -> None:
        return self.correct.float() / self.total
