"""Pytorch Lightning implementation for approximated VanillaTransformer."""

import logging
from argparse import ArgumentParser
from typing import Dict, Union

import torch
from torch import Tensor

from ....approximation.controller import ModelApproximationController
from ....models.vanilla_transformer.model import VanillaTransformer
from ..language_model_for_generation import LitApproximatedLanguageModelForGeneration

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LitApproximatedVanillaTransformer(LitApproximatedLanguageModelForGeneration):
    """Pytorch lightning model for the approximated VanillaTransformer."""

    def __init__(
        self,
        model: VanillaTransformer,
        controller: ModelApproximationController,
        model_args: Dict[str, Union[float, int, str]],
    ) -> None:
        """Construct an LM lightning module.

        Args:
            model_args: model's arguments.
        """

        if not isinstance(model, VanillaTransformer):
            raise TypeError(
                f"The model you are trying to approximate is not of class '{VanillaTransformer}'. Build a specific PyTorch lightning model inheriting from 'LitApproximatedModel'."
            )

        super().__init__(
            model=model,
            controller=controller,
            model_args=model_args,
        )

        self.cumulative_accuracy: Tensor = torch.tensor(0.0)

    def validation_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        """Validation step which encompasses the forward pass and the computation of the accuracy and the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            accuracy and loss computed on the batch.
        """

        loss = self.model(**batch).loss  # type:ignore
        self.log("val_loss", loss, prog_bar=True)

        # NOTE: testing is done with a batch size of 1

        input_ids = batch["encoder_input_ids"]
        labels = batch["decoder_input_ids"]

        # generating the predicted sequence
        predictions = self.model.generate(  # type: ignore
            input_ids,
            do_sample=False,
            max_length=self.model_args["max_length"],
            num_beams=self.model_args["num_beams"],
        )

        ground_truth = labels[~batch["decoder_padding_mask"]]

        # check if the prediction perfectly matches the labels
        accuracy = (
            torch.tensor(1.0)
            if torch.equal(predictions.view(-1), ground_truth.view(-1))
            else torch.tensor(0.0)
        )
        self.log("val_accuracy", accuracy)

        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_test_epoch_start(self) -> None:
        self.cumulative_accuracy = torch.tensor(0.0)
        return super().on_test_epoch_start()

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # type: ignore
        """
        Test step which encompasses the forward pass and the computation of the accuracy and the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            accuracy computed on the batch.
        """

        # NOTE: testing is done with a batch size of 1

        input_ids = batch["encoder_input_ids"]
        labels = batch["decoder_input_ids"]

        # generating the predicted sequence
        predictions = self.model.generate(  # type: ignore
            input_ids,
            do_sample=False,
            max_length=self.model_args["max_length"],
            num_beams=self.model_args["num_beams"],
        )

        ground_truth = labels[~batch["decoder_padding_mask"]]

        # check if the prediction perfectly matches the labels
        accuracy = (
            torch.tensor(1.0)
            if torch.equal(predictions.view(-1), ground_truth.view(-1))
            else torch.tensor(0.0)
        )
        self.log("test_accuracy", accuracy)

        self.cumulative_accuracy = self.cumulative_accuracy + accuracy

        return {"test_accuracy": accuracy}

    def return_results_metrics(self, **kwargs) -> Dict[str, float]:
        """Returns the evaluation metrics.

        Args:
            support: number of samples tested.
            **kwargs: additional keyword arguments that might be passed, unused.

        Returns:
            A dictionary containing the test accuracy.
        """
        support: int = kwargs["support"]
        return {"test_accuracy": self.cumulative_accuracy.item() / support}

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

        # model configuration arguments
        parser.add_argument("--embedding_dim", type=int, default=256)
        parser.add_argument("--ffnn_hidden_dim", type=int, default=2048)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--num_attention_heads", type=int, default=8)
        parser.add_argument("--num_encoder_layers", type=int, default=4)
        parser.add_argument("--num_decoder_layers", type=int, default=4)
        parser.add_argument("--attention_mask_value", type=float, default=float("-inf"))
        parser.add_argument("--init_std", type=float, default=0.02)
        parser.add_argument("--max_position_embeddings", type=int, default=5000)
        parser.add_argument("--num_beams", type=int, default=1)

        return parser
