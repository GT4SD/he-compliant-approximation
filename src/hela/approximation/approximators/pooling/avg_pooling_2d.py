"""Module approximator for pooling."""

from typing import Any, Dict, List

from torch import Tensor, nn

from ..core import ModuleApproximator


class AvgPooling2dApproximator(ModuleApproximator):
    """Handles the approximation of the pooling layer.

    Attributes:
        supported_layer_types: contains the classes of the modules or functions that the approximator can approximate.
        approximation_type: name to identify the approximator referring to the type of approximation module.
        is_approximation_trainable: establishes if the approximation contain some trainable parameters.
    """

    supported_layer_types = {
        nn.MaxPool2d,
    }
    approximation_type = "avg_pooling_2d"
    is_approximation_trainable = False

    def __init__(
        self, parameters: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the AvgPooling2dApproximator.

        Args:
            parameters: parameters of the AvgPooling2dApproximation modules. Defaults to {}.
        """
        super().__init__(parameters, **kwargs)
        self.approximations: List[AvgPooling2dApproximation] = []

    def approximate_module(
        self, model: nn.Module, id: str, pretrained: bool, **kwargs: Dict[str, Any]
    ) -> nn.Module:
        """Approximates the module identified by the id.

        Args:
            model: model that contains the module to be approximated.
            id: identifier of the module to be approximated.
            pretrained: specifies which kind of module approximation should be returned: trainable or pretrained version.

        Returns:
            approximated module.
        """

        # retrieving the multihead module that is going to be approximated
        kwargs = {"pooling": getattr(model, id)}
        if pretrained:
            return self.get_pretrained_approximation(module=getattr(model, id))
        else:
            return self.get_trainable_approximation(**kwargs)

    def get_trainable_approximation(self, **kwargs: Dict[str, Any]) -> nn.Module:
        """Approximates the module for the training phase.

        Returns:
            approximated module ready for the training phase.
        """
        original_max_pooling_2d: nn.MaxPool2d = kwargs["pooling"]  # type: ignore

        self.parameters["kernel_size"] = self.parameters.get(
            "kernel_size", original_max_pooling_2d.kernel_size
        )
        self.parameters["stride"] = self.parameters.get(
            "stride", original_max_pooling_2d.stride
        )
        self.parameters["padding"] = self.parameters.get(
            "padding", original_max_pooling_2d.padding
        )
        self.parameters["ceil_mode"] = self.parameters.get(
            "ceil_mode", original_max_pooling_2d.ceil_mode
        )
        self.parameters["count_include_pad"] = True
        self.parameters["divisor_override"] = None

        new_approximation = AvgPooling2dApproximation(**self.parameters)
        # adding the module to the approximation list
        self.approximations.append(new_approximation)
        return new_approximation

    def get_pretrained_approximation(
        self, module: nn.Module, **kwargs: Dict[str, Any]
    ) -> nn.Module:
        """Converts the trainable approximation of the module into its pretrained form.

        Args:
            module: module approximation to be converted.

        Raises:
            ValueError: this method must be called for AvgPooling2dApproximation modules.

        Returns:
            approximated module in its pretrained form.
        """
        if not isinstance(module, AvgPooling2dApproximation):
            raise ValueError(f"{module.__class__} is not a {AvgPooling2dApproximation}")
        return module


class AvgPooling2dApproximation(nn.Module):
    """Average pooling over 2 dimensions.

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = nn.MaxPool2d

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """Initializes the AvgPooling2dApproximation.

        Args:
            **kwargs: Additional keyword arguments to pass to nn.AvgPool2d constructor.
        """
        super().__init__()

        self.is_trainable = False
        self.avg_pooling = nn.AvgPool2d(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        """Performs average pooling.

        Args:
            input: input values.

        Returns:
            pooled input.
        """
        return self.avg_pooling(input)
