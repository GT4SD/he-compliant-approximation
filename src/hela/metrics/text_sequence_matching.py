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
        """Updates the metric for a single prediction.

        Args:
            preds: prediction tensor. Expected to have a single dimension.
            target: target tensor. Expected to have a single dimension.
        """
        self.correct += torch.equal(preds, target)
        self.total += torch.tensor(1)

    def compute(self) -> Tensor:
        return self.correct.float() / self.total
