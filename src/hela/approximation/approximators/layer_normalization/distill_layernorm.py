"""Module approximator for layer normalization modules."""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from ....models.vanilla_transformer.model import ExtendedLayerNorm
from ..core import ModuleApproximator


class DistillLayerNormApproximator(ModuleApproximator):
    """Handles the approximation of ReLU and GeLU activation functions.

    Attributes:
        supported_layer_types: contains the classes of the modules or functions that the approximator can approximate.
        approximation_type: name to identify the approximator referring to the type of approximation module.
        is_approximation_trainable: establishes if the approximation contain some trainable parameters.
    """

    supported_layer_types = {nn.LayerNorm, ExtendedLayerNorm}
    approximation_type = "distill_layernorm"
    is_approximation_trainable = True

    def __init__(
        self, parameters: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the DistillLayerNormApproximator.

        Args:
            parameters: parameters of the DistillLayerNormApproximation modules. Defaults to {}.
        """
        super().__init__(parameters, **kwargs)
        self.approximations: List[
            Union[PairedLayerNorm, DistillLayerNormApproximation]
        ] = []

    def get_trainable_approximation(self, **kwargs: Dict[str, Any]) -> nn.Module:
        """Approximates the module for the training phase.

        Returns:
            approximated module ready for the training phase.
        """
        new_approximation = PairedLayerNorm(
            layernorm=kwargs["layernorm"],  # type: ignore
            **self.parameters,
        )
        # adding the module to the approximation list
        self.approximations.append(new_approximation)
        return new_approximation

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
        # retrieving the layernorm module that is going to be approximated
        original_layernorm = {"layernorm": getattr(model, id)}
        if pretrained:
            setattr(
                model,
                id,
                self.get_pretrained_approximation(
                    module=original_layernorm["layernorm"]
                ),
            )
        else:
            setattr(
                model,
                id,
                self.get_trainable_approximation(**original_layernorm),
            )
        return getattr(model, id)

    def get_pretrained_approximation(
        self, module: nn.Module, **kwargs: Dict[str, Any]
    ) -> nn.Module:
        """Converts the trainable approximation of the module into its pretrained form.

        Args:
            module: module approximation to be converted.

        Raises:
            ValueError: this method must be called for PairedLayerNorm modules.

        Returns:
            approximated module in its pretrained form.
        """
        if not isinstance(module, PairedLayerNorm):
            raise ValueError(f"{module.__class__} is not a {PairedLayerNorm}")
        # cloning the approximated module
        approximated_layernorm = deepcopy(module.approximated_layernorm)
        # updating the approximations list
        if module in self.approximations:
            self.approximations[
                self.approximations.index(module)
            ] = approximated_layernorm
        return approximated_layernorm

    def configure_optimizers(self, lightning_model: pl.LightningModule) -> Optimizer:
        """Configures a custom optimizer to distill the layer normalization module knowledge inside the approximation.

        Args:
            lightning_model: pytorch lightning model.

        Returns:
            custom optimizer for the pytorch lightning model.
        """
        # freezing other model parameters
        for param in lightning_model.model.parameters():  # type: ignore
            param.requires_grad = False

        # enabling the training for the parameters of the layernorm approximation
        learnable_parameters = []
        for module in self.approximations:
            learnable_parameters.extend(
                list(module.approximated_layernorm.parameters())  # type: ignore
            )
            for param in module.approximated_layernorm.parameters():  # type: ignore
                param.requires_grad = True

        # defining of the optimizer
        optimizer = optim.AdamW(
            params=learnable_parameters,
            lr=lightning_model.model_args["learning_rate"],  # type: ignore
            betas=(lightning_model.model_args["adam_beta1"], lightning_model.model_args["adam_beta2"]),  # type: ignore
            eps=lightning_model.model_args["adam_epsilon"],  # type: ignore
            weight_decay=lightning_model.model_args["adam_weight_decay"],  # type: ignore
        )
        return optimizer

    def training_step_loss(
        self, loss: Tensor, lightning_model: pl.LightningModule
    ) -> Tensor:
        """Sets the loss to distill the knowledge of the layer normalization inside its approximation.
        Follows the implementation suggested in [THE-X: Privacy-Preserving Transformer Inference with Homomorphic Encryption](https://aclanthology.org/2022.findings-acl.277.pdf).

        Args:
            loss: value of the loss at the current training step.
            lightning_model: pytorch lightning model.

        Returns:
            custom loss to distill the knowledge of the layer normalization inside its approximation.
        """
        loss = self.approximations[0].step_loss  # type: ignore
        for module in self.approximations[1:]:
            loss = loss + module.step_loss
        return loss


class DistillLayerNormApproximation(nn.Module):
    """Linear layer approximation of layer normalization.
    Follows the implementation suggested in [THE-X: Privacy-Preserving Transformer Inference with Homomorphic Encryption](https://aclanthology.org/2022.findings-acl.277.pdf)

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = nn.LayerNorm

    def __init__(self, normalized_shape: Union[int, List, torch.Size] = 256) -> None:
        """Initializes the DistillLayerNormApproximation.

        Args:
            normalized_shape: dimension of the linear layer (should match the embedding dimension). Defaults to 256.
        """
        super().__init__()

        self.normalized_shape = normalized_shape

        self.weights = nn.Parameter(
            torch.ones(self.normalized_shape), requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        """Linear layer approximation of layer normalization.

        Args:
            input: input values.

        Returns:
            approximated value of layer normalization.
        """
        return self.weights * input + self.bias


class PairedLayerNorm(nn.Module):
    """Module to apply distillation to learn a linear transformation that approximates the layer normalization.

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = nn.LayerNorm

    def __init__(
        self, layernorm: nn.LayerNorm, normalized_shape: Optional[int] = None
    ) -> None:
        """Initializes the PairedLayerNorm.

        Args:
            layernorm: layer normalization module that is going to be approximated.
            normalized_shape: dimension of the linear layer (should match the embedding dimension). Defaults to None.
        """
        super().__init__()

        kwargs: Dict[str, Any] = {}
        if normalized_shape is None:
            kwargs["normalized_shape"] = layernorm.normalized_shape
        else:
            kwargs["normalized_shape"] = normalized_shape

        self.original_layer_norm = layernorm
        self.approximated_layernorm = DistillLayerNormApproximation(**kwargs)

        # containing another LayerNorm module inside we want to stop the substitution
        self.allow_recursive_search = False
        self.is_trainable = True

        self.step_loss: Tensor = Tensor([0.0])

    def forward(self, input: Tensor) -> Tensor:
        """Forwards the input through the layer normalization and the approximation module and computes the MSE loss between the two outputs.

        Args:
            input: input values.

        Returns:
            normalized values through layer normalization.
        """
        original_layernorm_output = self.original_layer_norm(input)
        approximated_layernorm_output = self.approximated_layernorm(input)

        self.step_loss = nn.MSELoss()(
            approximated_layernorm_output, original_layernorm_output
        )

        return original_layernorm_output
