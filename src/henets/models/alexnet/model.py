"""AlexNet model."""

import torchvision
from torch import Tensor, nn

from .configuration import AlexNetConfig


class AlexNet(nn.Module):
    def __init__(self, config: AlexNetConfig) -> None:
        """"""
        super().__init__()
        self.config = config
        self.model = torchvision.models.AlexNet(
            num_classes=config.num_classes, dropout=config.dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = self.model(x)
        return nn.functional.softmax(x, dim=1)
