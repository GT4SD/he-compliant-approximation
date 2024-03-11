"""Pytorch Lightning modules implementation for approximated models."""

import logging
from argparse import ArgumentParser
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchmetrics
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
        """Constructs the lightning module for approximated models.

        Args:
            model:
            controller:
            model_args:
        """
        super().__init__()

        self.model = model
        self.controller = controller
        self.model_args = model_args

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Implement forward method.")

    def update_model(
        self,
        new_model: nn.Module,
        new_controller: ModelApproximationController,
    ) -> None:
        """Updates the model and the controller used for the approximation.

        Args:
            new_model:
            new_controller:
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

    def _perform_approximators_on_train_epoch_start(self) -> None:
        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    approximator.on_train_epoch_start(epoch=self.current_epoch)

    def on_train_epoch_start(self) -> None:
        self._perform_approximators_on_train_epoch_start()

    def training_step(self, *args, **kwargs) -> Union[Tensor, Dict[str, Any]]:
        raise NotImplementedError("Implement training_step method.")

    def _perform_approximators_on_train_epoch_end(self) -> None:
        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    approximator.on_train_epoch_end(
                        epoch=self.current_epoch, save_dir=self.logger.log_dir  # type: ignore
                    )

    def on_train_epoch_end(self) -> None:
        self._perform_approximators_on_train_epoch_end()

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


class LitApproximatedTransformer(LitApproximatedModel):
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

        if not self.controller.to_approximate.modules_set == set():
            for approximator in set(self.controller.approximators.values()):
                if approximator.is_approximation_trainable:
                    loss = approximator.training_step_loss(
                        loss=loss, lightning_model=self
                    )

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


class LitApproximatedCNN(LitApproximatedModel):
    """Pytorch lightning model for the approximated CNN."""

    def __init__(
        self,
        model: nn.Module,
        controller: ModelApproximationController,
        model_args: Dict[str, Union[float, int, str]],
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

    def configure_optimizers(
        self,
    ) -> Dict[str, Optimizer]:
        """Creates and returns the optimizer.

        Returns:
            output (dict of str: Any):
                - optimizer: the optimizer used to update the parameter.
        """

        optimizer = super().configure_optimizers()["optimizer"]

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
        logits = self(features)
        loss = self.loss(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
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
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Union[Tensor, torchmetrics.Accuracy]]:
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

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
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

    def on_test_epoch_end(self) -> None:
        # definition of the test accuracy value for the creation of result.json
        # NOTE: this is needed otherwise the self.test_accuracy.compute() performed in
        #       return_results_metrics would raise an error having the update history
        #       clear.
        self.test_accuracy_value = float(self.test_accuracy.compute().cpu().numpy())
        return super().on_test_epoch_end()

    def return_results_metrics(self, **kwargs: Dict[str, Any]) -> Dict[str, float]:
        """Returns the evaluation metrics.

        Args:
            **kwargs: additional keyword arguments that might be passed, unused.

        Returns:
            A dictionary containing the test accuracy.
        """
        return {"test_accuracy": self.test_accuracy_value}

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
        # pretrained model loading arguments
        parser.add_argument("--model_name_or_path", type=str, default=None)
        parser.add_argument("--model_config_name", type=str, default=None)

        return parser
