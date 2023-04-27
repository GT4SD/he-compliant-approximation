"""Module approximator for not scaled attention query key product."""

from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from ...core import ModuleApproximator
from ...multihead.customizable_multihead import _scaled_dot_product


class NotScaledQueryKeyDotProductApproximator(ModuleApproximator):
    """Handles the approximation of the query-key product in a multihead attention module.

    Attributes:
        supported_layer_types: contains the classes of the modules or functions that the approximator can approximate.
        approximation_type: name to identify the approximator referring to the type of approximation module.
        is_approximation_trainable: establishes if the approximation contain some trainable parameters.
    """

    supported_layer_types = {_scaled_dot_product}
    approximation_type = "not_scaled"
    is_approximation_trainable = False

    def __init__(
        self, parameters: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the NotScaledQueryKeyDotProductApproximator.

        Args:
            parameters: parameters of the NotScaledQueryKeyDotProduct modules. Defaults to {}.
        """
        super().__init__(parameters, **kwargs)
        self.approximations: List[NotScaledQueryKeyDotProduct] = []

    def get_trainable_approximation(self, **kwargs: Any) -> nn.Module:
        """Approximates the module for the training phase.

        Returns:
            approximated module ready for the training phase.
        """
        new_approximation = NotScaledQueryKeyDotProduct()
        # adding the module to the approximation list
        self.approximations.append(new_approximation)
        return new_approximation

    def get_pretrained_approximation(
        self, module: nn.Module, **kwargs: Any
    ) -> nn.Module:
        """Converts the trainable approximation of the module into its pretrained form.

        Args:
            module: module approximation to be converted.

        Raises:
            ValueError: this method must be called for NotScaledQueryKeyDotProduct modules.

        Returns:
            approximated module in its pretrained form.
        """
        if not isinstance(module, NotScaledQueryKeyDotProduct):
            raise ValueError(
                f"{module.__class__} is not a {NotScaledQueryKeyDotProduct}"
            )
        return module


class NotScaledQueryKeyDotProduct(nn.Module):
    """Not scaled query-key dot product for attention.

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = _scaled_dot_product

    def __init__(self) -> None:
        """Initializes the NotScaledQueryKeyDotProduct."""
        super().__init__()

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        """Not scaled query-key dot product.

        Args:
            query: attention query values.
            key: attention key values.

        Returns:
            not scaled dot product between query and key matrices.
        """
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        return torch.bmm(query, key.transpose(-2, -1))
