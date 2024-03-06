"""Pytorch Lightning implementation for approximated LeNet."""

import logging
from argparse import ArgumentParser
from typing import Dict, Optional, Tuple, Union, Any

import torch
import torchmetrics
from torch import Tensor

from ....approximation.controller import ModelApproximationController
from ....models.lenet.model import LeNet
from .core import LitApproximatedModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LitApproximatedLeNet(LitApproximatedModel):
    """Pytorch lightning model for the approximated LeNet."""

    def __init__(
        self,
        model: LeNet,
        controller: ModelApproximationController,
        model_args: Dict[str, Union[float, int, str]],
    ) -> None:
        """Construct a LeNet lightning module.

        Args:
            model_args: model's arguments.
        """

        if not isinstance(model, LeNet):
            raise TypeError(
                f"The model you are trying to approximate is not of class '{LeNet}'. Build a specific PyTorch lightning model inhereting from 'LitApproximatedModel'."
            )

        super().__init__(
            model=model,
            controller=controller,
            model_args=model_args,
        )
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.loss = torch.nn.functional.cross_entropy

    def forward(self, input: Tensor) -> Tensor:
        """Forwards the input through the model.

        Args:
            input: tensor to be forwarded.

        Return:
            model's output
        """
        return self.model(input)

    def _compute_loss(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the loss for the given batch of data.

        Parameters:
            batch: a tuple containing features and true labels.

        Returns:
            A tuple containing the loss value, true labels, and predicted labels.
        """
        features, true_labels = batch
        logits = self(features)
        loss = self.loss(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        """Training step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the inputs and the labels.
            batch_idx: index of the current batch, unused.

        Returns:
            loss computed on the batch.
        """

        loss, true_labels, predicted_labels = self._compute_loss(batch)

        # to account for Dropout behavior during evaluation
        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._compute_loss(batch)
        self.train_accuracy.update(predicted_labels, true_labels)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True, on_step=False)
        self.model.train()

        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    loss = approximator.training_step_loss(
                        loss=loss, lightning_model=self
                    )

        self.log("train_loss", loss)

        return loss

    def validation_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Optional[Dict[str, Tensor]]:
        """Validation step which encompasses the forward pass and the computation of the accuracy and the loss value.

        Args:
            batch: dictionary containing the inputs and the labels.
            batch_idx: index of the current batch, unused.

        Returns:
            accuracy and loss computed on the batch.
        """
        loss, true_labels, predicted_labels = self._compute_loss(batch)

        self.log("valid_loss", loss)
        self.val_accuracy(predicted_labels, true_labels)
        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return {"val_loss": loss, "val_accuracy": self.val_accuracy}
    
    def return_results_metrics(self, **kwargs: Dict[str, Any]) -> Dict[str, float]:
        """Returns the evaluation metrics.

        Args:
            **kwargs: additional keyword arguments that might be passed, unused.

        Returns:
            A dictionary containing the test accuracy.
        """
        return {"test_accuracy": self.test_accuracy}

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Test step which encompasses the forward pass and the computation of the accuracy and the loss value.

        Args:
            batch: dictionary containing the inputs and the labels.
            batch_idx: index of the current batch, unused.

        Returns:
            accuracy computed on the batch.
        """
        loss, true_labels, predicted_labels = self._compute_loss(batch)

        self.test_accuracy(predicted_labels, true_labels)
        self.log("test_accuracy", self.test_accuracy, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.model_args["learning_rate"]  # type: ignore
        )
        return {"optimizer": optimizer}

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
        parser.add_argument("--model_name_or_path", type=str, default=None)
        parser.add_argument("--model_config_name", type=str, default=None)

        # model configuration arguments
        parser.add_argument("--lenet_type", type=str, default="LeNet-5")
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--greyscale", type=bool, default=True)

        return parser
