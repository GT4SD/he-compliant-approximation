"""Pytorch Lightning modules implementation for approximated language models for text generation tasks."""

import logging
from argparse import ArgumentParser
from typing import Dict, Optional

import torch.optim as optim
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .core import LitApproximatedModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LitApproximatedLanguageModelForGeneration(LitApproximatedModel):
    """Pytorch lightning module for the approximation of a transformer model."""

    def configure_optimizers(
        self,
    ) -> Dict[str, Optimizer]:
        """Creates and returns the optimizer.

        Returns:
            output (dict of str: Any):
                - optimizer: the optimizer used to update the parameter.
        """

        optimizer = self._configure_approximators_optimizers()

        if optimizer is None:
            # definition of the default optimizer
            optimizer = optim.AdamW(
                params=self.parameters(),
                lr=self.model_args["learning_rate"],  # type: ignore
                betas=(self.model_args["adam_beta1"], self.model_args["adam_beta2"]),  # type: ignore
                eps=self.model_args["adam_epsilon"],  # type: ignore
                weight_decay=self.model_args["adam_weight_decay"],  # type: ignore
            )

        return {"optimizer": optimizer}

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

        loss = self._trainable_approximations_training_step_loss_computation(loss=loss)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Optional[Dict[str, Tensor]]:  # type: ignore
        """
        Validation step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            loss computed on the batch.
        """
        loss = self.model(**batch).loss  # type:ignore

        # TODO: should I add the loss value of the approximations to the validation step?
        # maybe the approximation should have a flag that tells if shoud do something with the loss value during the validation step

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

        # default optimizer configuration arguments
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.98)
        parser.add_argument("--adam_epsilon", type=float, default=1e-9)
        parser.add_argument("--adam_weight_decay", type=float, default=0.01)
        # pretrained model loading arguments
        parser.add_argument("--model_name_or_path", type=str, default=None)
        parser.add_argument("--model_config_name", type=str, default=None)

        return parser
