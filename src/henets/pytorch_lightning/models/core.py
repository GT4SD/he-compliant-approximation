"""Pytorch Lightning modules implementation for approximated models."""

from argparse import ArgumentParser
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from ...approximation.controller import ModelApproximationController
from ...utils.logging import setup_logger

logger = setup_logger(__name__, logging_level="info")


class LitApproximatedModel(pl.LightningModule):
    """Pytorch lightning module for the approximation of a model."""

    def __init__(
        self,
        model: nn.Module,
        controller: ModelApproximationController,
        model_args: Dict[str, Union[float, int, str]],
        **kwargs,
    ) -> None:
        """Constructs the lightning module for approximated models.

        Args:
            model:
            controller:
            model_args:
        """
        super().__init__(**kwargs)

        self.model = model
        self.controller = controller
        self.model_args = model_args

    def forward(self, *args, **kwargs) -> Any:
        """Forwards the input through the model."""
        raise NotImplementedError(f"Implement forward method for {self.__class__}.")

    def update_model(
        self,
        new_model: nn.Module,
        new_controller: ModelApproximationController,
    ) -> None:
        """Updates the model and the controller used for the approximation.

        Args:
            new_model: new model to be used for the approximation.
            new_controller: new controller to be used for the approximation.
        """
        self.model = new_model
        self.controller = new_controller

    def _configure_approximators_optimizers(self) -> Optional[Optimizer]:

        optimizer: Optional[Optimizer] = None
        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    optimizer = approximator.configure_optimizers(lightning_model=self)
        if optimizer is not None:
            logger.warning(
                "The optimizer has been set depending on one of the approximators. Check for the presence of conflicts in each step."
            )
        return optimizer

    def configure_optimizers(
        self,
    ) -> Dict[str, Optimizer]:
        """Create and return the optimizer.

        Returns:
            output (dict of str: Any):
                - optimizer: the optimizer used to update the parameter.
        """
        raise NotImplementedError("Implement configure_optimizers method.")

    def _approximators_on_train_epoch_start(self) -> None:
        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    approximator.on_train_epoch_start(epoch=self.current_epoch)

    def on_train_epoch_start(self) -> None:
        self._approximators_on_train_epoch_start()

    def training_step(self, *args, **kwargs) -> Union[Tensor, Dict[str, Any]]:
        raise NotImplementedError("Implement training_step method.")

    def _trainable_approximations_training_step_loss_computation(
        self, loss: Tensor
    ) -> Tensor:
        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    loss = approximator.training_step_loss(
                        loss=loss, lightning_model=self
                    )
        return loss

    def _approximators_on_train_epoch_end(self) -> None:
        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    approximator.on_train_epoch_end(
                        epoch=self.current_epoch, save_dir=self.logger.log_dir  # type: ignore
                    )

    def on_train_epoch_end(self) -> None:
        self._approximators_on_train_epoch_end()

    def validation_step(
        self, *args, **kwargs
    ) -> Optional[Union[Tensor, Dict[str, Any]]]:
        raise NotImplementedError("Implement validation_step method.")

    def test_step(self, *args, **kwargs) -> Optional[Union[Tensor, Dict[str, Any]]]:
        raise NotImplementedError("Implement test_step method.")

    def return_results_metrics(self, **kwargs) -> Dict[str, float]:
        raise NotImplementedError(
            "Implement return_results_metric method to print the testing results in a json file."
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds model specific arguments to the parser.

        Args:
            parent_parser: argument parser.

        Returns:
            updated parser.
        """
        raise NotImplementedError(
            "Implement add_model_specific_args to be able to add arguments to the argument parser."
        )
