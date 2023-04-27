"""ModuleApproximator abstract class."""

import logging
import os
from typing import Any, Callable, Dict, Optional, Set, Type, Union

import pytorch_lightning as pl
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ModuleApproximator:
    """Handles the approximation of some modules or functions.

    Attributes:
        supported_layer_types: contains the classes of the modules or functions that the approximator can approximate.
        approximation_type: name to identify the approximator referring to the type of approximation module.
        is_approximation_trainable: establishes if the approximation contain some trainable parameters.
    """

    supported_layer_types: Set[Union[Type[nn.Module], Callable]]

    approximation_type: str
    is_approximation_trainable: bool

    def __init__(
        self, parameters: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the module approximator.

        Args:
            parameters: parameters of the approximation handled by the approximator. Defaults to {}.
        """
        self.parameters = parameters

    def approximate_module(
        self,
        model: nn.Module,
        id: str,
        pretrained: bool,
        **kwargs: Dict[str, Any],
    ) -> nn.Module:
        """Approximates the module identified by the id.

        Args:
            model: model that contains the module to be approximated.
            id: identifier of the module to be approximated.
            pretrained: specifies which kind of module approximation should be returned: trainable or pretrained version.

        Returns:
            approximated module.
        """

        if pretrained:
            return self.get_pretrained_approximation(module=getattr(model, id))  # type: ignore
        else:
            return self.get_trainable_approximation()

    def get_trainable_approximation(self, **kwargs: Dict[str, Any]) -> nn.Module:
        """Approximates the module for the training phase.

        Raises:
            NotImplementedError: this method must be defined in the child classes.

        Returns:
            approximated module ready for the training phase.
        """
        raise NotImplementedError(
            f"Inherit ModuleApproximator to implement a trainable approximation for module={self.supported_layer_types}."
        )

    def get_pretrained_approximation(
        self, module: nn.Module, **kwargs: Dict[str, Any]
    ) -> nn.Module:
        """Converts the trainable approximation of the module into its pretrained form.

        Args:
            module: module approximation to be converted.

        Raises:
            NotImplementedError: this method must be defined in the child classes.

        Returns:
            approximated module in its pretrained form.
        """
        raise NotImplementedError(
            f"Inherit ModuleApproximator to implement a pretrained approximation for module={self.supported_layer_types}."
        )

    def configure_optimizers(
        self, lightning_model: pl.LightningModule
    ) -> Optional[Optimizer]:
        """Configures a custom optimizer for the pytorch lightning model.

        Args:
            lightning_model: pytorch lightning model.

        Returns:
            custom optimizer.
        """
        logger.warning(
            f"Override 'configure_optimizers' of ModuleApproximator to configure an custom optimizer for a trainable approximation of module={self.supported_layer_types}."
        )
        return None

    def on_train_epoch_start(self, epoch: int = 0) -> None:
        """Performs actions at the start of each training epoch.

        Args:
            epoch: number of the current starting epoch. Defaults to 0.
        """
        logger.warning(
            f"Override 'on_train_epoch_start' of ModuleApproximator to perform actions on train epoch start for a trainable approximation of module={self.supported_layer_types}."
        )

    def training_step_loss(
        self, loss: Tensor, lightning_model: pl.LightningModule
    ) -> Tensor:
        """Performs computations on the loss at the end of each training step.

        Args:
            loss: value of the loss at the current training step.
            lightning_model: pytorch lightning model.

        Returns:
            custom loss value.
        """
        logger.warning(
            f"Override 'training_step_loss' of ModuleApproximator to perform computations on the loss at the end of each training step for a trainable approximation of module={self.supported_layer_types}."
        )
        return loss

    def on_train_epoch_end(self, epoch: int = 0, save_dir: str = os.getcwd()) -> None:
        """Performs actions at the end of each training epoch.

        Args:
            epoch: number of the current ending epoch. Defaults to 0.
            save_dir: path of the current training pipeline step directory (inside the experiment directory). Defaults to os.getcwd().
        """
        logger.warning(
            f"Override 'on_train_epoch_start' of ModuleApproximator to perform actions on train epoch start for a trainable approximation of module={self.supported_layer_types}."
        )
