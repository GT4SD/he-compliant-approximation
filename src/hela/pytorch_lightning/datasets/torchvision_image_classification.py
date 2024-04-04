"""Torchvision image classification datasets."""

import logging
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TORCHVISION_DATASETS = {
    "mnist": {
        "dataset_class": datasets.MNIST,
        "train_length": 55000,
        "val_length": 5000,
        "transform": transforms.Compose([transforms.ToTensor()]),
        "image_size": 32,
        "grayscale": True,
        "num_classes": 10,
    },
    "fashion_mnist": {
        "dataset_class": datasets.FashionMNIST,
        "train_length": 55000,
        "val_length": 5000,
        "transform": transforms.Compose([transforms.ToTensor()]),
        "image_size": 32,
        "grayscale": True,
        "num_classes": 10,
    },
    "cifar10": {
        "dataset_class": datasets.CIFAR10,
        "train_length": 45000,
        "val_length": 5000,
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        ),
        "image_size": 32,
        "grayscale": False,
        "num_classes": 10,
    },
}

DATASET_ARGS = {
    "dataset_name": {
        "type": str,
        "required": False,
        "default": "mnist",
    },
    "dataset_path": {
        "type": str,
        "required": False,
        "default": "./",
    },
    "image_size": {
        "type": int,
        "required": False,
        "default": None,
    },
    "batch_size": {
        "type": int,
        "required": False,
        "default": 32,
    },
    "num_dataloader_workers": {
        "type": int,
        "required": False,
        "default": 8,
    },
    "pin_memory": {
        "type": bool,
        "required": False,
        "default": False,
    },
    "persistent_workers": {
        "type": bool,
        "required": False,
        "default": False,
    },
}


class LitImageClassificationDataset(pl.LightningDataModule):

    def __init__(
        self,
        dataset_name: str = "mnist",
        dataset_path: str = "./",
        image_size: int = None,
        batch_size: int = 32,
        num_dataloader_workers: int = 8,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        """Initializes the data module.

        Args:
            dataset_name: dataset type to be used from TORCHVISION_DATASETS. Defaults to "mnist".
            dataset_path: path to the dataset. Defaults to "./".
            image_size: size of the images. Defaults to None.
            batch_size: batch size. Defaults to 32.
            num_dataloader_workers: number of workers for the dataloader. Defaults to 8.
            pin_memory: whether to pin memory or not. Defaults to False.
            persistent_workers: whether to keep the workers alive or not. Defaults to False.
        """
        super().__init__()

        dataset_name = str(dataset_name).lower()
        assert TORCHVISION_DATASETS.get(
            dataset_name, None
        ), "The dataset {dataset_name} is not available."

        self.dataset = TORCHVISION_DATASETS[dataset_name]
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        cpus_count = os.cpu_count()
        if cpus_count is not None:
            self.num_dataloader_workers = min(num_dataloader_workers, cpus_count)
        else:
            self.num_dataloader_workers = num_dataloader_workers

        self.image_size = image_size
        if self.image_size is None:
            self.image_size = self.dataset["image_size"]

    def prepare_data(self) -> None:
        """Downloads and applies the transformations to the dataset images."""

        self.dataset_path = os.path.join(self.dataset_path, "datasets")
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        self.dataset["dataset_class"](root=self.dataset_path, download=True)

        # generating the values transformation
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                self.dataset["transform"],
            ]
        )

    def load(self) -> None:
        """Loads train, validation and test datasets."""

        self.prepare_data()

        train = self.dataset["dataset_class"](
            root=self.dataset_path,
            train=True,
            transform=self.transform,
            download=False,
        )

        test = self.dataset["dataset_class"](
            root=self.dataset_path,
            train=False,
            transform=self.transform,
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
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the dataloader for the validation step.

        Returns:
            pytorch dataloader.
        """
        return DataLoader(
            dataset=self.datasets["validation"],
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates the dataloader for the test step.

        Returns:
            pytorch dataloader.
        """
        return DataLoader(
            dataset=self.datasets["test"],
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def get_num_classes(self) -> int:
        return self.dataset["num_classes"]

    def is_grayscale(self) -> bool:
        return self.dataset["grayscale"]

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments for the dataset to the parser.

        Args:
            parent_parser: argument parser to be updated.

        Returns:
            updated parser.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for arg in DATASET_ARGS:
            arg_params = DATASET_ARGS[arg]
            if arg_params["type"] == bool:
                parser.add_argument(
                    f"--{arg}",
                    dest=arg,
                    action="store_true",
                    default=arg_params["default"],
                )
            else:
                parser.add_argument(
                    f"--{arg}",
                    type=arg_params["type"],
                    required=arg_params["required"],
                    default=arg_params["default"],
                )

        return parser
