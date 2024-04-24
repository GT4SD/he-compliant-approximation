"""AlexNet configuration."""


class AlexNetConfig:
    """AlexConfig implementation."""

    model_type = "AlexNet"

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.5,
        **kwargs,
    ) -> None:
        """Initialization of the configuration.

        Args:
            num_classes: number of classes of the problem. Defaults to 10.
            dropout: value of the dropout neurons. Defaults to 0.5.
        """

        # model hyperparameters configuration
        self.num_classes = num_classes
        self.dropout = dropout
