"""Module approximator for polynomial softmax."""

from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from ..core import ModuleApproximator


class PolynomialSoftmax(nn.Module):
    """Polynomial version of the softmax function.

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = nn.Softmax

    def __init__(
        self,
        order: int = 2,
        dim: int = -1,
        skip_normalization: bool = False,
        init_alpha: float = 1e-3,
        norm_loss_weight: float = 1.,
    ) -> None:
        """Initializes the PolynomialSoftmax.

        Args:
            order: order of the polynomial. Defaults to 2.
            dim: dimension along which the polynomial softmax is computed. Defaults to -1.
            min_max_normalization: whether to apply min-max normalization instead of normalizing with the sum of outputs. Defaults to False.
            skip_normalization: whether to avoid normalization. Defaults to False.
        """
        super().__init__()

        self.is_trainable = True

        # the polynomial to be substituted to exp of softmax must be positive definite (i.e. only even orders)
        assert order % 2 == 0 and order > 0
        self.order = order
        self.dim = dim

        self.skip_normalization = skip_normalization
        self.norm_loss_weight = norm_loss_weight

        if not skip_normalization:
            self._forward_function = self._sum_normalization_forward
        else:
            self._forward_function = self._skip_normalization_forward
            self.alpha = nn.Parameter(torch.ones(1) * init_alpha, requires_grad=True)
            self.step_loss: Tensor = Tensor([0.0])

    def _skip_normalization_forward(self, input: Tensor) -> Tensor:
        """Polynomial softmax without normalization.

        Args:
            input: input values.

        Returns:
            not normalized output.
        """
        input = input * self.alpha
        input = torch.pow(input, self.order)
        # computing the training step loss from the sum of the output values
        sum_of_input = input.sum(dim=self.dim)
        self.step_loss = self.norm_loss_weight * nn.MSELoss()(sum_of_input, torch.ones_like(sum_of_input))
        return input

    def _sum_normalization_forward(self, input: Tensor) -> Tensor:
        """Polynomial softmax with normalization through the sum of outputs.

        Args:
            input: input values.

        Returns:
            normalized output.
        """
        input.sum()
        input = torch.pow(input, self.order)
        return input * torch.reciprocal((input.sum(dim=self.dim, keepdim=True) + 1e-5))

    def forward(self, input: Tensor) -> Tensor:
        """Polynomial softmax.

        Args:
            input: input values.

        Returns:
            `self.order` polynomial softmax (with only the highest order term) of the input.
        """
        return self._forward_function(input)


class PolynomialSoftmaxApproximator(ModuleApproximator):
    """Handles the approximation of the softmax function in a multihead attention module.

    Attributes:
        supported_layer_types: contains the classes of the modules or functions that the approximator can approximate.
        approximation_type: name to identify the approximator referring to the type of approximation module.
        is_approximation_trainable: establishes if the approximation contain some trainable parameters.
    """

    supported_layer_types = {
        nn.Softmax,
        PolynomialSoftmax,  # this is needed to allow the additional loss term to be present also in later training steps
    }
    approximation_type = "polynomial"
    is_approximation_trainable = True

    def __init__(
        self, parameters: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the PolynomialSoftmaxApproximator.

        Args:
            parameters: parameters of the PolynomialSoftmax modules. Defaults to {}.
        """
        super().__init__(parameters, **kwargs)
        self.approximations: List[PolynomialSoftmax] = []

    def approximate_module(
        self, model: nn.Module, id: str, pretrained: bool, **kwargs: Any
    ) -> nn.Module:
        """Approximates the module identified by the id.

        Args:
            model: model that contains the module to be approximated.
            id: identifier of the module to be approximated.
            pretrained: specifies which kind of module approximation should be returned: trainable or pretrained version.

        Returns:
            approximated module.
        """

        # retrieving the softmax module that is going to be approximated
        kwargs = {"softmax": getattr(model, id)}
        if pretrained:
            setattr(
                model,
                id,
                self.get_pretrained_approximation(module=getattr(model, id)),
            )
        else:
            setattr(
                model,
                id,
                self.get_trainable_approximation(**kwargs),
            )
        return getattr(model, id)

    def get_trainable_approximation(self, **kwargs: Dict[str, Any]) -> nn.Module:
        """Approximates the module for the training phase.

        Returns:
            approximated module ready for the training phase.
        """

        softmax: Union[nn.Softmax, PolynomialSoftmax] = kwargs["softmax"]  # type: ignore

        
        if isinstance(softmax, PolynomialSoftmax):
            # adding the module to the approximation list
            self.approximations.append(softmax)
            return softmax
        
        new_approximation = PolynomialSoftmax(**self.parameters)
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
            ValueError: this method must be called for PolynomialSoftmax modules.

        Returns:
            approximated module in its pretrained form.
        """
        if not isinstance(module, PolynomialSoftmax):
            raise ValueError(f"{module.__class__} is not a {PolynomialSoftmax}")
        return module

    def training_step_loss(
        self, loss: Tensor, lightning_model: pl.LightningModule
    ) -> Tensor:
        """Adds an additional term to force the sum on the last dimension of the output to be as close as possible to 1.
        This is done if the approximated module is not normalizing its output.

        Args:
            loss: value of the loss at the current training step.
            lightning_model: pytorch lightning model.

        Returns:
            custom loss value.
        """
        add_loss: Optional[Tensor] = None
        for module in self.approximations:
            if module.skip_normalization:
                if module == self.approximations[0]:
                    add_loss = module.step_loss
                else:
                    add_loss = add_loss + module.step_loss
        if add_loss is not None:
            lightning_model.log("CE_loss", loss.clone().detach(), prog_bar=True)
            lightning_model.log("SoftmaxNorm_loss", add_loss.clone().detach(), prog_bar=True)
            loss = loss + add_loss
            # considering that the objective is the sum to 1: we would have additional loss of 120 (*1e-4= 0.012) and at the beginning 50000 (*1e-4= 6)
            # the weight of the loss is considered to obtain a loss that is equal in magnitude compared to the original loss that ranges roughly between 6.5 and 0.015
        return loss
