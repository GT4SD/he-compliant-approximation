"""LeNet model."""

from torch import Tensor, nn

from .configuration import LeNetConfig


class LeNet(nn.Module):
    """LeNet implementation."""

    def __init__(self, config: LeNetConfig) -> None:
        """Initializes a LeNet.

        Args:
            config: configuration of the LeNet.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.config = config

        for conv_index in range(config.num_conv):
            self.layers.append(
                nn.Conv2d(
                    config.conv_in_channels[conv_index],
                    config.conv_out_channels[conv_index],
                    kernel_size=config.conv_kernel_size,
                    stride=config.conv_stride,
                    padding=config.conv_padding,
                )
            )
            self.layers.append(nn.BatchNorm2d(config.conv_out_channels[conv_index]))
            self.layers.append(nn.ReLU())
            self.layers.append(
                nn.MaxPool2d(
                    kernel_size=config.pool_kernel_size, stride=config.pool_stride
                )
            )

        for linear_index in range(config.num_linear):
            self.layers.append(
                nn.Linear(
                    config.linear_in_features[linear_index],
                    config.linear_out_features[linear_index],
                )
            )
            if linear_index < config.num_linear - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        """Implements a forward pass through the LeNet.

        Args:
            x: input given to the LeNet.

        Returns:
            output generated by the LeNet.
        """
        for layer in self.layers:
            if (
                isinstance(layer, nn.Linear)
                and x.size(1) == self.config.conv_out_channels[-1]
            ):
                x = x.reshape(x.size(0), -1)
            x = layer(x)
        return x
