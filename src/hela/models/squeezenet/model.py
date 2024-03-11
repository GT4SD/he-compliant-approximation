"""SqueezeNet model."""

import torchvision
from torch import Tensor, nn

from .configuration import SqueezeNetConfig


class SqueezeNet(nn.Module):
    def __init__(self, config: SqueezeNetConfig) -> None:
        """"""
        super().__init__()
        self.model = torchvision.models.SqueezeNet(**config)

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return self.model(x)
