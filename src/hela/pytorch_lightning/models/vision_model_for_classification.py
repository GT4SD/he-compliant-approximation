"""Pytorch Lightning modules implementation for approximated vision models for classification tasks."""

from argparse import ArgumentParser
from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from ...approximation.controller import ModelApproximationController
from ...utils.logging import setup_logger
from .core import LitApproximatedModel

logger = setup_logger(__name__, logging_level="info")


class LitApproximatedVisionModelForClassification(LitApproximatedModel):
    """Approximated vision model for classification tasks."""

    def __init__(
        self,
        model: nn.Module,
        controller: ModelApproximationController,
        model_args: Dict[str, Union[float, int, str]],
        metrics: MetricCollection = MetricCollection(
            [MulticlassAccuracy(num_classes=10)]
        ),
        **kwargs,
    ) -> None:
        """Construct a CNN lightning module.

        Args:
            model:
            controller:
            model_args: model's arguments.
        """

        super().__init__(
            model=model,
            controller=controller,
            model_args=model_args,
            **kwargs,
        )
        self.loss = nn.CrossEntropyLoss()
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, input: Tensor) -> Tensor:
        """Forwards the input through the model.

        Args:
            input: tensor to be forwarded.

        Return:
            model's output
        """
        return self.model(input)

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
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.model_args["learning_rate"]  # type: ignore
            )
        return {"optimizer": optimizer}

    def _compute_loss(
        self, batch: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the loss for the given batch of data.

        Parameters:
            batch: a tuple containing features and true labels.

        Returns:
            A tuple containing the loss value, true labels, and predicted labels.
        """
        features, true_labels = batch
        logits = self.model(features)
        loss = self.loss(logits, true_labels)

        return loss, true_labels, logits

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the inputs and the labels.
            batch_idx: index of the current batch, unused.

        Returns:
            loss computed on the batch.
        """

        loss, true_labels, logits = self._compute_loss(batch)

        # to account for Dropout behavior during evaluation
        self.model.eval()
        with torch.no_grad():
            features, true_labels = batch
            logits = self.model(features)
        self.model.train()

        output = self.train_metrics(logits, true_labels)
        self.log_dict(output, on_epoch=True, on_step=True)

        loss = self._trainable_approximations_training_step_loss_computation(loss=loss)
        self.log("train_loss", loss)

        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        """Validation step which encompasses the forward pass and the computation of the accuracy and the loss value.

        Args:
            batch: dictionary containing the inputs and the labels.
            batch_idx: index of the current batch, unused.

        Returns:
            accuracy and loss computed on the batch.
        """
        loss, true_labels, logits = self._compute_loss(batch)

        output = self.valid_metrics(logits, true_labels)
        self.log_dict(output, on_epoch=True, on_step=False)

        return {"val_loss": loss}

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step which encompasses the forward pass and the computation of the accuracy and the loss value.

        Args:
            batch: dictionary containing the inputs and the labels.
            batch_idx: index of the current batch, unused.

        Returns:
            accuracy computed on the batch.
        """
        _, true_labels, logits = self._compute_loss(batch)

        output = self.test_metrics(logits, true_labels)
        self.log_dict(output, on_epoch=True, on_step=False)

    def return_results_metrics(self, **kwargs: Dict[str, Any]) -> Dict[str, float]:
        """Returns the evaluation metrics.

        Args:
            **kwargs: additional keyword arguments that might be passed, unused.

        Returns:
            A dictionary containing the test accuracy.
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
        group = parser.add_argument_group("vision_model_for_classification")

        # default optimizer configuration arguments
        group.add_argument("--learning_rate", type=float, default=0.0001)

        return parser
