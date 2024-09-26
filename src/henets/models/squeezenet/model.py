"""SqueezeNet model."""

import torchvision
from torch import Tensor, nn

from .configuration import SqueezeNetConfig


class SqueezeNet(nn.Module):
    def __init__(self, config: SqueezeNetConfig) -> None:
        """"""
        super().__init__()
        self.config = config
        self.model = torchvision.models.SqueezeNet(
            version=config.version,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = self.model(x)
        return nn.functional.softmax(x, dim=1)
