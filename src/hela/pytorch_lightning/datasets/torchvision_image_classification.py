"""Torchvision image classification datasets."""

import logging
import os
from argparse import ArgumentParser
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

torchvision_datasets = {
    "mnist": {
        "dataset_class": datasets.MNIST,
        "train_length": 55000,
        "val_length": 5000,
    },
    "fashion_mnist": {
        "dataset_class": datasets.FashionMNIST,
        "train_length": 55000,
        "val_length": 5000,
    },
    "cifar10": {
        "dataset_class": datasets.CIFAR10,
        "train_length": 45000,
        "val_length": 5000,
    },
}


class LitImageClassificationDataset(pl.LightningDataModule):

    def __init__(self, dataset_args: Dict[str, Any]) -> None:
        """Initializes the data module.

        Args:
            dataset_args: dictionary containing the arguments for the lightning data module creation.
        """
        super().__init__()

        self.dataset_args = dataset_args

        dataset_type = str(self.dataset_args["dataset_type"]).lower()
        assert torchvision_datasets.get(
            dataset_type, None
        ), "The dataset {dataset_type} is not available."
        self.dataset = torchvision_datasets[dataset_type]

        cpus_count = os.cpu_count()
        if cpus_count is not None:
            self.dataset_args["num_dataloader_workers"] = min(
                self.dataset_args["num_dataloader_workers"], cpus_count
            )

    def prepare_data(self) -> None:
        """Downloads and applies the transformations to the dataset images."""

        self.dataset_args["data_path"] = os.path.join(
            self.dataset_args["data_path"], "datasets"
        )
        if not os.path.exists(self.dataset_args["data_path"]):
            os.mkdir(self.dataset_args["data_path"])

        self.dataset["dataset_class"](
            root=self.dataset_args["data_path"], download=True
        )

        # resizing and normalizing values
        self.resize_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.dataset_args["image_size"], self.dataset_args["image_size"])
                ),
                transforms.ToTensor(),
            ]
        )

    def load(self) -> None:
        """Loads train, validation and test datasets."""

        self.prepare_data()

        train = self.dataset["dataset_class"](
            root=self.dataset_args["data_path"],
            train=True,
            transform=self.resize_transform,
            download=False,
        )

        test = self.dataset["dataset_class"](
            root=self.dataset_args["data_path"],
            train=False,
            transform=self.resize_transform,
            download=False,
        )

        train, val = random_split(
            train,
            lengths=[self.dataset["train_length"], self.dataset["val_length"]],
            generator=torch.Generator().manual_seed(42),
        )

        self.datasets = {
            "train": train,
            "validation": val,
            "test": test,
        }

        logger.info(
            f"Training set size: {len(self.datasets['train'])} - Validation set size: {len(self.datasets['validation'])} - Test set size: {len(self.datasets['test'])}"  # type: ignore
        )

    def train_dataloader(self) -> DataLoader:
        """Creates the dataloader for the training step.

        Returns:
            pytorch dataloader.
        """
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.dataset_args["batch_size"],
            drop_last=True,
            shuffle=True,
            num_workers=self.dataset_args["num_dataloader_workers"],
            pin_memory=self.dataset_args["pin_memory"],
            persistent_workers=self.dataset_args["persistent_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the dataloader for the validation step.

        Returns:
            pytorch dataloader.
        """
        return DataLoader(
            dataset=self.datasets["validation"],
            batch_size=self.dataset_args["batch_size"],
            drop_last=False,
            shuffle=False,
            num_workers=self.dataset_args["num_dataloader_workers"],
            pin_memory=self.dataset_args["pin_memory"],
            persistent_workers=self.dataset_args["persistent_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        """Creates the dataloader for the test step.

        Returns:
            pytorch dataloader.
        """
        return DataLoader(
            dataset=self.datasets["test"],
            batch_size=self.dataset_args["batch_size"],
            drop_last=False,
            shuffle=False,
            num_workers=self.dataset_args["num_dataloader_workers"],
            pin_memory=self.dataset_args["pin_memory"],
            persistent_workers=self.dataset_args["persistent_workers"],
        )

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments for the dataset to the parser.

        Args:
            parent_parser: argument parser to be updated.

        Returns:
            updated parser.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--dataset_type", type=str, default="mnist")
        parser.add_argument("--data_path", type=str, default="./")
        parser.add_argument("--num_dataloader_workers", type=int, default=8)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--image_size", type=int, default=32)
        parser.add_argument(
            "--pin_memory", dest="pin_memory", action="store_true", default=False
        )
        parser.add_argument(
            "--persistent_workers",
            dest="persistent_workers",
            action="store_true",
            default=False,
        )

        return parser
