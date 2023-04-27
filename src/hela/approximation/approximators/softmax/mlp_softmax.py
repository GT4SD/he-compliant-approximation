"""Module approximator for MLP approximation of softmax function."""

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import default_data_collator

from ..core import ModuleApproximator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MLPSoftmaxApproximator(ModuleApproximator):
    """Handles the approximation of the softmax function in a multihead attention module.

    Attributes:
        supported_layer_types: contains the classes of the modules or functions that the approximator can approximate.
        approximation_type: name to identify the approximator referring to the type of approximation module.
        is_approximation_trainable: establishes if the approximation contain some trainable parameters.
    """

    supported_layer_types = {nn.Softmax}
    approximation_type = "MLP_softmax"
    is_approximation_trainable = False

    def __init__(
        self, parameters: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the MLPSoftmaxApproximator.

        Args:
            parameters: parameters of the MLPSoftmaxApproximation modules. Defaults to {}.
        """
        super().__init__(parameters, **kwargs)

        self.approximations: List[MLPSoftmaxApproximation] = []

        # removing the argument passed in the tests to avoid the layer training
        self.is_unit_test = self.parameters.pop("unit_test", None)

        approximation = MLPSoftmaxApproximation(
            **self.parameters,
        )

        self.approximated_softmax: MLPSoftmaxApproximation = approximation
        self.is_pretrained = False
        self.save_path: str

    def pretraining_approximation(self):
        """Pretrains the weights of the approximation and then freeze them."""
        if self.is_unit_test is None and not self.is_pretrained:
            # training the approximate module
            softmax_dataset = LitMLPSoftmaxDataset()

            softmax_dataset.load()
            train_dataloader = softmax_dataset.train_dataloader()
            val_dataloader = softmax_dataset.val_dataloader()

            # defining the pytorch lightning model
            lightning_model = LitMLPSoftmaxApproximation(
                model=self.approximated_softmax
            )

            callbacks: List = []
            callbacks.append(
                ModelCheckpoint(
                    save_last=True,
                )
            )
            logger = TensorBoardLogger(
                save_dir=self.save_path, name=None, version="softmax_approximation"
            )

            trainer = pl.Trainer(
                max_epochs=1,
                callbacks=callbacks,
                logger=logger,
                limit_val_batches=0,
                accelerator="auto",
                deterministic=True,
            )

            trainer.fit(
                lightning_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            # the weights should be saved to be loaded whe we get a new trainable approximation
            self.approximated_softmax = lightning_model.model

        # disbling training to keep the pre-trained weights
        for param in self.approximated_softmax.parameters():
            param.requires_grad = False

    def approximate_module(
        self, model: nn.Module, id: str, pretrained: bool, **kwargs: Any
    ) -> nn.Module:
        """Approximates the module identified by the id.

        Args:
            model: model that contains the module to be approximated.
            id: identifier of the module to be approximated.
            pretrained: specifies which kind of module approximation should be returned: trainable or pretrained version.

        Returns:
            approximated module.
        """
        # updating the saving path to the directory of the current pipeline step
        self.save_path = kwargs.get("save_path", os.getcwd())

        if not self.is_pretrained:
            # pretraining the approximation module before substitution (if needed)
            if os.path.exists(os.path.join(self.save_path, "softmax_approximation")):
                ckpt_dir = os.path.join(
                    self.save_path,
                    "softmax_approximation",
                    "checkpoints",
                )
                if not os.path.exists(ckpt_dir):
                    print("WARNING: pretraining softmax since no checkpoint was found.")
                    self.pretraining_approximation()
                else:
                    lit_checkpoint = torch.load(
                        os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
                    )
                    self.approximated_softmax.load_state_dict(
                        lit_checkpoint["state_dict"]
                    )
            else:
                self.pretraining_approximation()

            self.is_pretrained = True
            # disabling training of the parameters to keep the pretrained weights
            for param in self.approximated_softmax.parameters():
                param.requires_grad = False

        if pretrained:
            return self.get_pretrained_approximation(module=getattr(model, id))  # type: ignore
        else:
            return self.get_trainable_approximation()

    def get_trainable_approximation(self, **kwargs: Any) -> nn.Module:
        """Approximates the module for the training phase.

        Returns:
            approximated module ready for the training phase.
        """
        # creating a new instance of the approximated softmax
        new_approximation = deepcopy(self.approximated_softmax)
        # adding the module to the approximation list
        self.approximations.append(new_approximation)

        return new_approximation

    def get_pretrained_approximation(
        self, module: nn.Module, **kwargs: Any
    ) -> nn.Module:
        """Converts the trainable approximation of the module into its pretrained form.

        Args:
            module: module approximation to be converted.

        Raises:
            ValueError: this method must be called for MLPSoftmaxApproximation modules.

        Returns:
            approximated module in its pretrained form.
        """
        if not isinstance(module, MLPSoftmaxApproximation):
            raise ValueError(f"{module.__class__} is not a {MLPSoftmaxApproximation}")
        return module


class ReciprocalApproximation(nn.Module):
    """MLP to approximate the reciprocal of the sum.
    Follows the implementation suggested in [THE-X: Privacy-Preserving Transformer Inference with Homomorphic Encryption](https://aclanthology.org/2022.findings-acl.277.pdf).
    """

    def __init__(self, dim_size=16):
        """Initializes the ReciprocalApproximation.

        Args:
            dim_size: dimension of the MLP hidden layer. Defaults to 16.
        """
        super().__init__()
        self.transform = nn.Linear(1, dim_size)
        self.dense = nn.Linear(dim_size, dim_size)
        self.predict = nn.Linear(dim_size, 1)
        self.activation = nn.ReLU()

        for _, module in self.named_modules():
            self._init_weights(module=module)

    def forward(self, x: Tensor) -> Tensor:
        """Approximates the reciprocal of the input value.

        Args:
            x: input value representing the sum of the output of the preceding module.

        Returns:
            multiplicative factor that approximates the reciprocal of the sum (i.e. approximates 1/x).
        """
        x = self.activation(self.transform(x))
        x = self.activation(self.dense(x))
        x = self.predict(x)
        return x

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes the weights.

        Args:
            module: layer of the model.
        """

        if isinstance(module, nn.Linear):
            # initilaization of the weights of a linear module
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class MLPSoftmaxApproximation(nn.Module):
    """Softmax approximation through MLP.
    Follows the implementation suggested in [THE-X: Privacy-Preserving Transformer Inference with Homomorphic Encryption](https://aclanthology.org/2022.findings-acl.277.pdf).

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = nn.Softmax

    def __init__(self, dim_size=16):
        """Initializes the MLPSoftmaxApproximation.

        Args:
            dim_size: dimension of the ReciprocalApproximation hidden layer. Defaults to 16.
        """
        super().__init__()

        self.reciprocal = ReciprocalApproximation(dim_size=dim_size)
        self.activation = nn.ReLU()
        self.dim = -1

    def forward(self, input: Tensor) -> Tensor:
        """MLP softmax approximation.

        Args:
            input: input values.

        Returns:
            approximated normalized values.
        """
        # input: [E, Nt, Ns]
        # input transformation to obtain a better approximation of the exp in softmax
        t = input / 3 + 2

        # computing the aproximation of the exponential
        # exp_of_score: [E, Nt, Ns]
        exp_of_score = self.activation(t * t * t)

        # computing the sum of exponentials
        # normalization_term: [E, Nt, 1, 1]
        normalization_term = exp_of_score.sum(self.dim, keepdim=True).unsqueeze(-1)

        # computing the approximation of the reciprocal of the sum of exponentials
        # normalization_term: [E, Nt, 1]
        normalization_term = self.reciprocal(normalization_term).squeeze(dim=self.dim)

        # computing the approximation of the normalized output
        # output: [E, Nt, Ns]
        return exp_of_score * normalization_term


"""Pytorch Lightning implementation for MLPSoftmaxApproximation."""


class MLPSoftmaxDataset(Dataset):
    """MLPSoftmaxApproximation training dataset."""

    def __init__(
        self,
    ) -> None:
        """Initializes the dataset."""
        self.length = 1000000

    def __len__(self) -> int:
        """Number of instances of the dataset.

        Returns:
           number of instances of the dataset.
        """
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Gets an item of the dataset.

        Args:
            index: index of the item.

        Returns:
            item containing a input for the MLP softmax approximation module.
        """

        # input and output of size: (B, Nt, Ns)
        # B: batch_size
        # B = 1
        # Nt: target sequence length
        Nt = 278
        # Ns: source sequence length
        Ns = 278

        item: Dict = {}
        # the input is taken as randomly distributed values following a normal distribution with mean 0 and standard deviation equal to 1
        # thus 99.7% of the values will be contained in [mean-2*sigma, mean+2*sigma] == [-3,3]
        item["input"] = torch.randn(Nt, Ns)

        return item


class LitMLPSoftmaxDataset(pl.LightningDataModule):
    """Pytorch-lightning data module for MLPSoftmaxApproximation."""

    def __init__(
        self,
    ) -> None:
        """Initializes the lightning data module for MLPSoftmaxApproximation training and validation."""

        super().__init__()

        self.datasets: Dict

        self.data_collator = default_data_collator

        self.dataset_args: Dict = {}
        self.dataset_args["batch_size"] = 256

        self.dataset_args["num_dataloader_workers"] = 16
        cpus_count = os.cpu_count()
        if cpus_count is not None:
            self.dataset_args["num_dataloader_workers"] = min(
                self.dataset_args["num_dataloader_workers"], cpus_count
            )

    def build_dataset(self) -> Dataset:
        """Builds the dataset.

        Returns:
            a torch Dataset.
        """
        return MLPSoftmaxDataset()

    def load(self) -> None:
        """Loads the train and validation datasets."""

        self.datasets = {
            "train": self.build_dataset(),
            "validation": self.build_dataset(),
        }

        logger.info(
            f"Training set size: {len(self.datasets['train'])} - Validation set size: {len(self.datasets['validation'])}"  # type: ignore
        )

    def train_dataloader(self) -> DataLoader:
        """Creates the DataLoader for the training step.

        Returns:
            pytorch dataloader.
        """
        return DataLoader(
            self.datasets["train"],  # type: ignore
            batch_size=self.dataset_args["batch_size"],
            num_workers=self.dataset_args["num_dataloader_workers"],
            collate_fn=self.data_collator,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the DataLoader for the validation step.

        Returns:
            pytorch dataloader.
        """
        return DataLoader(
            self.datasets["validation"],  # type: ignore
            batch_size=self.dataset_args["batch_size"],
            num_workers=self.dataset_args["num_dataloader_workers"],
            collate_fn=self.data_collator,
        )


class LitMLPSoftmaxApproximation(pl.LightningModule):
    """Pytorch lightning model for MLPSoftmaxApproximation."""

    def __init__(
        self,
        model: MLPSoftmaxApproximation,
    ) -> None:
        """Initializes a lightning module for the MLPSoftmaxApproximation.

        Args:
            model: MLPSoftmaxApproximation object to be trained and validated.
        """
        super().__init__()
        self.model = model
        # defining the ground truth function, i.e. Softmax
        self.softmax = nn.Softmax(dim=-1)
        self.loss_function = nn.MSELoss()

        self.model_args: Dict = {}
        self.model_args["learning_rate"] = 0.001

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        """Forward pass through the model.

        Raises:
            NotImplementedError: implement this function for the prediction step.
        """
        raise NotImplementedError(
            "Implement the forward function for the LitMLPSoftmaxApproximation."
        )

    def configure_optimizers(
        self,
    ) -> Dict[str, object]:
        """Creates the optimizer.

        Returns:
            output:
                - optimizer: the optimizer used to update the parameters.
        """

        # defining of the default optimizer
        optimizer = optim.Adam(
            params=self.parameters(),
            lr=self.model_args["learning_rate"],  # type: ignore
        )

        output = {
            "optimizer": optimizer,
        }

        return output  # type: ignore

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        """Training step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch.

        Returns:
            loss computed on the batch.
        """

        approx_output: Tensor = self.model(**batch)  # type:ignore
        softmax_output: Tensor = self.softmax(**batch)

        loss = self.loss_function(approx_output.view(-1), softmax_output.view(-1))

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # type: ignore
        """Validation step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch.

        Returns:
            output:
                - val_loss: validation loss computed on the batch.
        """
        approx_output = self.model(**batch)  # type:ignore
        softmax_output = self.softmax(**batch)

        loss = self.loss_function(approx_output.view(-1), softmax_output.view(-1))

        self.log("val_loss", loss)

        return {"val_loss": loss}
