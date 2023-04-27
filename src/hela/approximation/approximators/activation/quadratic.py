"""Module approximator for activation functions."""

from typing import Any, Dict, List

from torch import Tensor, nn

from ..core import ModuleApproximator


class QuadraticApproximator(ModuleApproximator):
    """Handles the approximation of ReLU activation function.

    Attributes:
        supported_layer_types: contains the classes of the modules or functions that the approximator can approximate.
        approximation_type: name to identify the approximator referring to the type of approximation module.
        is_approximation_trainable: establishes if the approximation contain some trainable parameters.
    """

    supported_layer_types = {
        nn.ReLU,
        nn.functional.relu,  # type: ignore
    }
    approximation_type = "quadratic"
    is_approximation_trainable = False

    def __init__(
        self, parameters: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the QuadraticApproximator.

        Args:
            parameters: parameters of the QuadraticApproximation modules. Defaults to {}.
        """
        super().__init__(parameters, **kwargs)
        self.approximations: List[QuadraticApproximation] = []

    def get_trainable_approximation(self, **kwargs: Dict[str, Any]) -> nn.Module:
        """Approximates the module for the training phase.

        Returns:
            approximated module ready for the training phase.
        """
        new_approximation = QuadraticApproximation()
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
            ValueError: this method must be called for QuadraticApproximation modules.

        Returns:
            approximated module in its pretrained form.
        """
        if not isinstance(module, QuadraticApproximation):
            raise ValueError(f"{module.__class__} is not a {QuadraticApproximation}")
        return module


class QuadraticApproximation(nn.Module):
    """Quadratic activation function.

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = nn.ReLU

    def __init__(self) -> None:
        """Initializes the QuadraticApproximation."""
        super().__init__()

        self.is_trainable = False

    def forward(self, input: Tensor) -> Tensor:
        """Quadratic activation function.

        Args:
            input: input values.

        Returns:
            square of the input.
        """
        return input * input
