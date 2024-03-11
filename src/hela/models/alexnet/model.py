"""AlexNet model."""

import torchvision
from torch import Tensor, nn

from .configuration import AlexNetConfig


class AlexNet(nn.Module):
    def __init__(self, config: AlexNetConfig) -> None:
        """"""
        super().__init__()
        self.model = torchvision.models.AlexNet(**config)

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return self.model(x)
