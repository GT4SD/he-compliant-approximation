"""Module approximator for layer normalization modules."""

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from ....models.vanilla_transformer.model import ExtendedLayerNorm
from ..core import ModuleApproximator


class LayerNormToBatchNormApproximator(ModuleApproximator):
    """Handles the approximation of layer normalization modules.

    Attributes:
        supported_layer_types: contains the classes of the modules or functions that the approximator can approximate.
        approximation_type: name to identify the approximator referring to the type of approximation module.
        is_approximation_trainable: establishes if the approximation contain some trainable parameters.
    """

    supported_layer_types = {nn.LayerNorm, ExtendedLayerNorm}
    approximation_type = "batchnorm"
    is_approximation_trainable = True

    def __init__(
        self, parameters: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the LayerNormToBatchNormApproximator.

        Args:
            parameters: parameters of the BatchNorm1dForTransformers modules. Defaults to {}.
        """
        super().__init__(parameters, **kwargs)

        self.approximations: List[BatchNorm1dForTransformers] = []

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

        # retrieving the num_features parameters from the layer normalization module that is going to be approximated
        kwargs = {"num_features": getattr(model, id).normalized_shape[0]}
        if pretrained:
            return self.get_pretrained_approximation(module=getattr(model, id))
            
        else:
            return self.get_trainable_approximation(**kwargs)

    def get_trainable_approximation(self, **kwargs: Dict[str, Any]) -> nn.Module:
        """Approximates the module for the training phase.

        Returns:
            approximated module ready for the training phase.
        """

        # setting up the num_features parameter of batch normalization module
        num_features = self.parameters.get("num_features", None)
        if num_features is None:
            self.parameters["num_features"] = kwargs["num_features"]
        else:
            self.parameters["num_features"] = num_features

        new_approximation = BatchNorm1dForTransformers(
            **self.parameters,
        )
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
            ValueError: this method must be called for BatchNorm1dForTransformers modules.

        Returns:
            approximated module in its pretrained form.
        """

        if not isinstance(module, BatchNorm1dForTransformers):
            raise ValueError(
                f"{module.__class__} is not a {BatchNorm1dForTransformers}"
            )
        return module

    def training_step_loss(
        self, loss: Tensor, lightning_model: pl.LightningModule
    ) -> Tensor:
        """Adds 2 additional terms that take into account the mean and the standard deviation Training Inference Discrepancy (TID) of batch normalization.
        The implementation follows "Understanding the Failure of Batch Normalization for Transformers in NLP" (https://arxiv.org/pdf/2210.05153.pdf).

        Args:
            loss: value of the loss at the current training step.
            lightning_model: pytorch lightning model.

        Returns:
            custom loss with 2 additional terms that take into account the mean and the standard deviation TID.
        """
        if self.parameters.get("regularized_BN", False):
            alpha = self.parameters.get("RBN_alpha", None)
            beta = self.parameters.get("RBN_beta", None)
            if alpha is None or beta is None:
                raise AttributeError(
                    f"Cannot apply Regularized Batch Normalization without alpha and beta parameters. Received alpha={alpha} , beta={beta}"
                )
            elif not isinstance(alpha, float) or not isinstance(beta, float):
                raise TypeError(
                    f"Expected 'RBN_alpha' and 'RBN_beta' to be float. Obtained type(alpha)={type(alpha)}, and type(beta)={type(beta)}"
                )
            for module in self.approximations:
                loss = (
                    loss
                    + alpha * torch.pow((module.norm.running_mean.data - module.norm.batch_mean.data).norm(p=2), 2)  # type: ignore
                    + beta * torch.pow((torch.sqrt(module.norm.running_var.data) - module.norm.batch_std_deviation.data).norm(p=2), 2)  # type: ignore
                )
        return loss


class BatchNorm1dForTransformers(nn.BatchNorm1d):
    """Batch normalization module adapted for transformers input dimensions.
    The channels/features are the last dimension, but torch.nn.BatchNorm1d needs them as dimension==1.
    To avoid continuos permutations a new class is defined and the forward method has been overrided.

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = nn.LayerNorm

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        """Initializes the BatchNorm1dForTransformers.

        Args:
            num_features: number of features of the input.
            eps: a value added to the denominator for numerical stability. Defaults to 1e-5.
            momentum: the value used for the running_mean and running_var computation.
                Can be set to None for cumulative moving average (i.e. simple average). Defaults to 0.1 .
            affine: a boolean value that when set to True, this module has learnable affine parameters. Defaults to True.
            track_running_stats: a boolean value that when set to True, this
                module tracks the running mean and variance, and when set to False,
                this module does not track such statistics, and initializes statistics
                buffers running_mean and running_var as None.
                When these buffers are None, this module always uses batch statistics.
                in both training and eval modes. Defaults to True.
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        # batch statistics for regularized batch normalization loss (taking into account TID)
        # they are None since they must be set only during training
        self.batch_mean: Optional[Tensor] = None
        self.batch_std_deviation: Optional[Tensor] = None

    def forward(self, input: Tensor) -> Tensor:
        """Created taking inspiration from https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
        and https://github.com/pytorch/pytorch/blob/31f311a816c026bbfca622d6121d6a7fab44260d/torch/nn/modules/batchnorm.py#L121
        """
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # computing running estimates
        if self.training:
            mean = input.mean([0, 1])
            # use biased variance during training
            var = input.var([0, 1], unbiased=False)
            n = input.numel() / input.size(2)
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean  # type: ignore
                )
                # update running_var with unbiased var
                self.running_var = (
                    exponential_average_factor * var * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_var  # type: ignore
                )
            # assigning batch statistics for regularized batch normalization loss
            # avoiding computations of gradients
            self.batch_mean = mean.detach().clone()
            self.batch_std_deviation = torch.sqrt(var).detach().clone()

        else:
            mean = self.running_mean  # type: ignore
            var = self.running_var  # type: ignore

        mean = mean.unsqueeze(0).unsqueeze(0)
        var = var.unsqueeze(0).unsqueeze(0)
        input = (input - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            input = input * self.weight.unsqueeze(0).unsqueeze(0) + self.bias.unsqueeze(
                0
            ).unsqueeze(0)

        return input
