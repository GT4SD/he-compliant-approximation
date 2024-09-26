"""LeNet configuration."""


class LeNetConfig:
    """LeNetConfig implementation."""

    model_type = "LeNet"

    def __init__(
        self,
        lenet_type: str = "LeNet-5",
        num_classes: int = 10,
        greyscale: bool = False,
        **kwargs,
    ) -> None:
        """Initialization of the configuration.

        Args:
            lenet_type: type of LeNet model. Defaults to "LeNet-5".
            num_classes: number of classes of the problem. Defaults to 10.
            greyscale: wether the input images are in greyscale. Defaults to True.
        """

        # model hyperparameters configuration
        self.num_classes = num_classes

        if greyscale:
            self.in_channels = 1
        else:
            self.in_channels = 3

        lenet_type = lenet_type.lower()
        if lenet_type == "lenet-1":
            self.input_size = [28, 28, self.in_channels]
            self.num_conv = 2
            self.conv_in_channels = [self.in_channels, 4]
            self.conv_out_channels = [4, 12]
            self.conv_kernel_size = 5
            self.conv_stride = 1
            self.conv_padding = 0
            self.pool_kernel_size = 2
            self.pool_stride = 2
            self.num_linear = 1
            self.linear_in_features = [12 * 4 * 4]
            self.linear_out_features = [num_classes]
        elif lenet_type == "lenet-4":
            self.input_size = [32, 32, self.in_channels]
            self.num_conv = 2
            self.conv_in_channels = [self.in_channels, 4]
            self.conv_out_channels = [4, 16]
            self.conv_kernel_size = 5
            self.conv_stride = 1
            self.conv_padding = 0
            self.pool_kernel_size = 2
            self.pool_stride = 2
            self.num_linear = 2
            self.linear_in_features = [400, 120]
            self.linear_out_features = [120, num_classes]
        elif lenet_type == "lenet-5":
            self.input_size = [32, 32, self.in_channels]
            self.num_conv = 2
            self.conv_in_channels = [self.in_channels, 6]
            self.conv_out_channels = [6, 16]
            self.conv_kernel_size = 5
            self.conv_stride = 1
            self.conv_padding = 0
            self.pool_kernel_size = 2
            self.pool_stride = 2
            self.num_linear = 3
            self.linear_in_features = [16 * 5 * 5, 120, 84]
            self.linear_out_features = [120, 84, num_classes]
        else:
            raise ValueError(
                f"The LeNet type {lenet_type} is not valid. Available: LeNet-1, LeNet-4, LeNet-5"
            )
        self.lenet_type = lenet_type
