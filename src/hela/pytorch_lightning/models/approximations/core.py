"""Pytorch Lightning module implementation for approximated models."""

import logging
from argparse import ArgumentParser
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch.optim as optim
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from ....approximation.controller import ModelApproximationController

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LitApproximatedModel(pl.LightningModule):
    """Pytorch lightning module for the approximation of a model."""

    def __init__(
        self,
        model: nn.Module,
        controller: ModelApproximationController,
        model_args: Dict[str, Union[float, int, str]],
    ) -> None:
        """Construct an LM lightning module.

        Args:

        """
        super().__init__()

        self.model = model
        self.controller = controller

        self.model_args = model_args

    def update_model(
        self,
        new_model: nn.Module,
        new_controller: ModelApproximationController,
    ) -> None:
        self.model = new_model
        self.controller = new_controller

    def configure_optimizers(
        self,
    ) -> Dict[str, object]:
        """Create and return the optimizer.

        Returns:
            output (dict of str: Any):
                - optimizer: the optimizer used to update the parameter.
        """

        optimizer: Optional[Optimizer] = None
        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    optimizer = approximator.configure_optimizers(lightning_model=self)

        if not isinstance(self.model_args["learning_rate"], float):
            raise ValueError("Learning rate should be float")

        if optimizer is None:
            # definition of the default optimizer
            optimizer = optim.AdamW(
                params=self.parameters(),
                lr=self.model_args["learning_rate"],  # type: ignore
                betas=(self.model_args["adam_beta1"], self.model_args["adam_beta2"]),  # type: ignore
                eps=self.model_args["adam_epsilon"],  # type: ignore
                weight_decay=self.model_args["adam_weight_decay"],  # type: ignore
            )
        else:
            logger.warning(
                "The optimizer has been set depending on one of the approximators. Check for the presence of conflicts in each step."
            )

        output = {
            "optimizer": optimizer,
        }

        return output  # type: ignore

    def on_train_epoch_start(self) -> None:
        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    approximator.on_train_epoch_start(epoch=self.current_epoch)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        """
        Training step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            loss computed on the batch.
        """
        loss = self.model(**batch).loss  # type:ignore

        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    loss = approximator.training_step_loss(
                        loss=loss, lightning_model=self
                    )

        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    approximator.on_train_epoch_end(
                        epoch=self.current_epoch, save_dir=self.logger.log_dir  # type: ignore
                    )

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # type: ignore
        """
        Validation step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            loss computed on the batch.
        """
        loss = self.model(**batch).loss  # type:ignore
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds model specific arguments to the parser.

        Args:
            parent_parser: argument parser.

        Returns:
            updated parser.
        """

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # training configuration arguments
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.98)
        parser.add_argument("--adam_epsilon", type=float, default=1e-9)
        parser.add_argument("--adam_weight_decay", type=float, default=0.01)
        parser.add_argument("--model_name_or_path", type=str, default=None)
        parser.add_argument("--model_config_name", type=str, default=None)

        return parser
