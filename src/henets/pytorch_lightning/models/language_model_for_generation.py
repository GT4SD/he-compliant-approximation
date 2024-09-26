"""Pytorch Lightning modules implementation for approximated language models for text generation tasks."""

from argparse import ArgumentParser
from typing import Any, Dict, Union

import torch.optim as optim
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection

from ...approximation.controller import ModelApproximationController
from ...metrics.text_sequence_matching import SequenceMatchingMetric
from ...utils.logging import setup_logger
from .core import LitApproximatedModel

logger = setup_logger(__name__, logging_level="info")


class LitApproximatedLanguageModelForGeneration(LitApproximatedModel):
    """Approximated language model for generation tasks."""

    def __init__(
        self,
        model: nn.Module,
        controller: ModelApproximationController,
        model_args: Dict[str, Union[float, int, str]],
        metrics: MetricCollection = MetricCollection([SequenceMatchingMetric()]),
        **kwargs,
    ) -> None:

        super().__init__(
            model=model,
            controller=controller,
            model_args=model_args,
            **kwargs,
        )
        self.valid_metrics = metrics.clone(prefix="valid_")
        self.test_metrics = metrics.clone(prefix="test_")

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

    def validation_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        """Validation step which encompasses the forward pass and the computation of the validation metrics and the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            validation metrics and loss computed on the batch.
        """

        loss = self.model(**batch).loss  # type:ignore
        self.log("val_loss", loss, prog_bar=True)

        # NOTE: validation is done with a batch size of 1

        input_ids = batch["encoder_input_ids"]
        labels = batch["decoder_input_ids"]

        # generating the predicted sequence
        predictions = self.model.generate(  # type: ignore
            input_ids,
            do_sample=False,
            max_length=self.model.config.max_length,
            num_beams=self.model.config.num_beams,
        )

        ground_truth = labels[~batch["decoder_padding_mask"]]

        # check if the prediction perfectly matches the labels
        valid_result = self.valid_metrics(
            preds=predictions.view(-1), target=ground_truth.view(-1)
        )
        self.log_dict(valid_result, on_epoch=True, on_step=False, prog_bar=True)

        return {"val_loss": loss}

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:  # type: ignore
        """
        Test step which encompasses the forward pass and the computation of the test metrics.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.
        """

        # NOTE: testing is done with a batch size of 1

        input_ids = batch["encoder_input_ids"]
        labels = batch["decoder_input_ids"]

        # generating the predicted sequence
        predictions = self.model.generate(  # type: ignore
            input_ids,
            do_sample=False,
            max_length=self.model.config.max_length,
            num_beams=self.model.config.num_beams,
        )

        ground_truth = labels[~batch["decoder_padding_mask"]]

        # check if the prediction perfectly matches the labels
        test_result = self.test_metrics(
            preds=predictions.view(-1), target=ground_truth.view(-1)
        )
        self.log_dict(test_result, on_epoch=True, on_step=False)

    def return_results_metrics(self, **kwargs: Dict[str, Any]) -> Dict[str, float]:
        """Returns the evaluation metrics.

        Args:
            **kwargs: additional keyword arguments that might be passed, unused.

        Returns:
            A dictionary containing the test metrics.
        """
        results = self.test_metrics.compute()
        for key, metric in results.items():
            results[key] = float(metric.cpu().numpy())
        return results

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
