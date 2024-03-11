"""SqueezeNet configuration."""


class SqueezeNetConfig:
    """SqueezeNetConfig implementation."""

    model_type = "SqueezeNet"

    def __init__(
        self,
        version: str = "1_0",
        num_classes: int = 10,
        dropout: float = 0.5,
    ) -> None:
        """Initialization of the configuration.

        Args:
            version: version of the model to be used (available: "1_0", "1_1"). Defaults to "1_0"
            num_classes: number of classes of the problem. Defaults to 10.
            dropout: value of the dropout neurons. Defaults to 0.5.
        """

        # model hyperparameters configuration
        self.version = version
        self.num_classes = num_classes
        self.dropout = dropout
