"""Module approximator for activation functions."""

from typing import Any, Dict, List, Union

import torch
from torch import Tensor, nn

from ..core import ModuleApproximator


class TrainableQuadraticApproximator(ModuleApproximator):
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
    approximation_type = "trainable_quadratic"
    is_approximation_trainable = True

    def __init__(
        self, parameters: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the TrainableQuadraticApproximator.

        Args:
            parameters: parameters of the TrainableQuadraticActivation modules. Defaults to {}.
        """
        super().__init__(parameters, **kwargs)

        self.current_number_of_epochs: int = 0
        self.approximations: List[Union[TrainableQuadraticActivation, PairedReLU]] = []

    def get_trainable_approximation(self, **kwargs: Dict[str, Any]) -> nn.Module:
        """Approximates the module for the training phase.

        Returns:
            approximated module ready for the training phase.
        """

        new_approximation = PairedReLU(**self.parameters)
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
            ValueError: this method must be called for TrainableQuadraticActivation modules.

        Returns:
            approximated module in its pretrained form.
        """
        if not isinstance(module, PairedReLU):
            raise ValueError(f"{module.__class__} is not a {PairedReLU}")

        # updating the approximation list
        if module in self.approximations:
            self.approximations[self.approximations.index(module)] = (
                module.approximated_relu
            )
        return module.approximated_relu

    def on_train_epoch_start(self, epoch: int = 0) -> None:
        """Updates the parameter (lambda_smooth_transition) used for the smooth transition process.

        Args:
            epoch: number of the current starting epoch. Defaults to 0.
        """
        # for each approximation we must update the number of epochs to allow the smooth transition
        # we add 1 since epochs starts from zero but a counter is needed (i.e. starting from 1)
        self.current_number_of_epochs = epoch + 1
        for module in self.approximations:
            # the smoothing lambda must be updated only when the smooth substitution is being performed
            # i.e. not during next steps of the training pipeline in which the activation is the TrainableQuadraticActivation
            if isinstance(module, PairedReLU):
                module.update_smoothing_lambda(self.current_number_of_epochs)  # type: ignore


class TrainableQuadraticActivation(nn.Module):
    """Trainable second order polynomial activation function.

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = nn.ReLU

    def __init__(self, input_dimension: int = 1) -> None:
        """Initializes the TrainableQuadraticActivation.

        Args:
            input_dimension: dimension of the weight parameters.
                If greater than 1 multiple polynomial will be learned,
                but must match the last input dimension. Defaults to 1.
        """
        super().__init__()

        # initializing the parameters to have a curve similar to a ReLU, i.e. y=x
        # following what is suggested in [A methodology for training homomorphicencryption friendly neural networks] (https://arxiv.org/abs/2111.03362)
        self.weight_power_2 = nn.Parameter(
            torch.zeros(input_dimension), requires_grad=True
        )
        self.weight_power_1 = nn.Parameter(
            torch.ones(input_dimension), requires_grad=True
        )

    def forward(self, input: Tensor) -> Tensor:
        """Second order polynomial activation function.

        Args:
            input: input values.

        Returns:
            second order polynomial (without constant value) of the input.
        """
        return torch.pow(self.weight_power_2 * input, 2) + self.weight_power_1 * input


class PairedReLU(nn.Module):
    """Approximation of ReLU activation, applying smoothing transition to learn a second order polynomial activation function.

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = nn.ReLU
    is_trainable = True
    # containing another ReLU module inside we want to stop the substitution
    allow_recursive_search = False

    def __init__(
        self,
        input_dimension: int = 1,
        smooth_factor: int = 50,
        warmup_epochs: int = 0,
    ) -> None:
        """Initializes the PairedReLU.

        Args:
            input_dimension: dimension of the weight parameters.
                If greater than 1 multiple polynomial will be learned,
                but must match the last input dimension. Defaults to 1.
            smooth_factor: number of epochs needed for smooth transition. Defaults to 50.
            warmup_epochs: number of epochs of training without smooth transition. Defaults to 0.
        """
        super().__init__()

        self.relu = nn.ReLU()
        self.approximated_relu = TrainableQuadraticActivation(
            input_dimension=input_dimension
        )
        # parameter used during the smooth transition to merge the output of the ReLU and the approximation
        self.lambda_smooth_transition: float = 0.0

        self.smooth_factor: int = smooth_factor
        self.warmup_epochs: int = warmup_epochs

    def forward(self, input: Tensor) -> Tensor:
        """Smooth transition between ReLU and a second order polynomial approximation.

        Args:
            input: input values.

        Returns:
            combination of ReLU and second order polynomial output, based on the smoothing parameter value.
        """
        return (1 - self.lambda_smooth_transition) * self.relu(
            input
        ) + self.lambda_smooth_transition * self.approximated_relu(input)

    def update_smoothing_lambda(self, epoch_number: int) -> None:
        """Updates the value of the smooth transition parameter.

        Args:
            epoch_number: current epoch number.
        """
        if epoch_number <= self.warmup_epochs:
            self.lambda_smooth_transition = 0
        elif epoch_number > self.warmup_epochs:
            if epoch_number < self.smooth_factor + self.warmup_epochs:
                self.lambda_smooth_transition = (
                    epoch_number - self.warmup_epochs
                ) / self.smooth_factor
            elif epoch_number >= self.smooth_factor + self.warmup_epochs:
                self.lambda_smooth_transition = 1
