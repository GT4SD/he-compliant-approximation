"""Module approximator for taylor softmax."""

from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from ..core import ModuleApproximator


class TaylorSoftmaxApproximator(ModuleApproximator):
    """Handles the approximation of the softmax function in a multihead attention module.

    Attributes:
        supported_layer_types: contains the classes of the modules or functions that the approximator can approximate.
        approximation_type: name to identify the approximator referring to the type of approximation module.
        is_approximation_trainable: establishes if the approximation contain some trainable parameters.
    """

    supported_layer_types = {
        nn.Softmax,
    }
    approximation_type = "taylor"
    is_approximation_trainable = False

    def __init__(
        self, parameters: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the TaylorSoftmaxApproximator.

        Args:
            parameters: parameters of the TaylorSoftmax modules. Defaults to {}.
        """
        super().__init__(parameters, **kwargs)
        self.approximations: List[TaylorSoftmax] = []

    def get_trainable_approximation(self, **kwargs: Dict[str, Any]) -> nn.Module:
        """Approximates the module for the training phase.

        Returns:
            approximated module ready for the training phase.
        """
        new_approximation = TaylorSoftmax(**self.parameters)
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
            ValueError: this method must be called for TaylorSoftmax modules.

        Returns:
            approximated module in its pretrained form.
        """
        if not isinstance(module, TaylorSoftmax):
            raise ValueError(f"{module.__class__} is not a {TaylorSoftmax}")
        return module


class TaylorSoftmax(nn.Module):
    """Taylor approximation of the softmax function.
    From [Exploring Alternatives to Softmax Function](https://arxiv.org/pdf/2011.11538.pdf).

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = nn.Softmax

    def __init__(self, order: int = 2, dim: int = -1) -> None:
        """Initializes the TaylorSoftmax.

        Args:
            order: order of the polynomial. Defaults to 2.
            dim: dimension along which the polynomial softmax is computed. Defaults to -1.
        """
        super().__init__()

        self.is_trainable = False

        # the polynomial to be substituted to exp of softmax must be positive definite (i.e. only even orders)
        assert order % 2 == 0 and order > 0
        self.order = order
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        """Taylor approzimation of softmax function of order `self.order` (with normalization through the sum of outputs)
        Taking inspiration from https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py

        Args:
            input: input values.

        Returns:
            normalized output.
        """
        value: Tensor = torch.ones_like(input)
        multiplicative_factor = 1.0
        for i in range(1, self.order + 1):
            multiplicative_factor *= i
            value = value + (input.pow(i) / multiplicative_factor)
        output = value / torch.sum(
            value, dim=self.dim, keepdim=True
        )  # no need of epsilon since the sum is always greater than 0 (always even order taylor expansions)
        return output
