"""MIT Smiles dataset routines-filtering, dataset building."""

import json
import logging
import os
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import PosixPath
from typing import Any, Dict, List, Union

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import default_data_collator
from transformers.tokenization_utils_base import BatchEncoding

from ...models.tokenizers.smiles import SmilesTokenizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SmilesDataset(Dataset):
    """Smiles dataset class."""

    def __init__(
        self,
        filepath: str,
        tokenizer: SmilesTokenizer,
        padding_idx: int = 0,
    ) -> None:
        """Initialize the LM data module.

        Args:
            filepath: path where the dataset is located.
            tokenizer: tokenize function to be used in the module.
        """

        self.filepath = filepath
        self.tokenizer = tokenizer
        self.length = SmilesDataset.count_examples(filepath)
        self.padding_idx = padding_idx

        if not self.filepath.endswith(".jsonl") and not self.filepath.endswith(".json"):
            raise ValueError(f"{filepath} is not a .jsonl or a json.")

    @lru_cache()
    def examples_reader(self) -> List[Dict[str, str]]:
        """Read instances from a filepath.

        Returns:
           list of instances.
        """
        with open(self.filepath) as fp:
            return [json.loads(line.strip()) for line in fp]

    @staticmethod
    def count_examples(filepath: str) -> int:
        """Count instances of a filepath.

        Args:
           filepath: path of the dataset.

        Returns:
           number of examples existed in the given filepath.
        """

        def _make_gen(reader):
            while True:
                b = reader(2**16)
                if not b:
                    break
                yield b

        with open(filepath, "rb") as f:
            count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))  # type: ignore
        return count

    def __len__(self) -> int:
        """Number of instances of the dataset.

        Returns:
           number of instances
        """
        return self.length

    def __getitem__(self, index) -> Dict:
        """Get an item of the dataset.

        Args:
            index: index of the item.

        Returns:
            tokenized item.
        """

        examples = self.examples_reader()
        example = examples[index]

        item = {}
        source_item = self.tokenizer(example["source"])
        target_item = self.tokenizer(example["target"])

        item["encoder_input_ids"] = source_item["input_ids"]
        item["encoder_padding_mask"] = source_item["attention_mask"].eq(
            self.padding_idx
        )

        item["decoder_input_ids"] = target_item["input_ids"]
        item["decoder_padding_mask"] = target_item["attention_mask"].eq(
            self.padding_idx
        )

        return item


class LitSmilesDataset(pl.LightningDataModule):
    """Pytorch-lightning-style data module for smiles dataset."""

    def __init__(
        self, dataset_args: Dict[str, Any], tokenizer: SmilesTokenizer
    ) -> None:
        """Initialize the data module.

        Args:
            dataset_args: dictionary containing the arguments for the lightning data module creation.
            tokenizer: tokenizer to be used in the module.
        """

        super().__init__()

        self.datasets: Dict

        self.dataset_args = dataset_args

        self.tokenizer = tokenizer

        self.data_collator = default_data_collator

        cpus_count = os.cpu_count()
        if cpus_count is not None:
            self.dataset_args["num_dataloader_workers"] = min(
                self.dataset_args["num_dataloader_workers"], cpus_count
            )

    def build_dataset(self, path: Union[str, PosixPath]) -> Dataset:
        """
        Builds the dataset.

        Args:
            path: path of the dataset or the directory that contains it.

        Returns:
            pytorch dataset.
        """
        path = str(path)
        if path.endswith(".jsonl") or path.endswith(".json"):
            return SmilesDataset(
                path, self.tokenize_function, padding_idx=self.tokenizer.pad_token_id
            )
        elif os.path.isdir(path):
            return ConcatDataset(
                datasets=[
                    SmilesDataset(
                        os.path.join(path, filename),
                        self.tokenize_function,
                        padding_idx=self.tokenizer.pad_token_id,
                    )
                    for filename in os.listdir(path)
                    if filename.endswith(".jsonl") or filename.endswith(".json")
                ]
            )
        else:
            raise TypeError(f"{path} type is not supported for dataset.")

    def tokenize_function(self, example: str) -> BatchEncoding:
        """Tokenizes the given examples.

        Args:
            examples: list of examples.

        Returns:
            tokenized examples.
        """

        truncation = self.dataset_args.get("truncation", True)
        padding = self.dataset_args.get("padding", "max_length")
        max_length = self.dataset_args.get("max_length", 278)
        return_token_type_ids = self.dataset_args.get("return_token_type_ids", False)

        return self.tokenizer(  # type: ignore
            example,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_token_type_ids=return_token_type_ids,
            return_tensors="pt",
        )

    def load(self) -> None:
        """Loads train, validation and test datasets from the given files."""

        self.datasets = {
            "train": self.build_dataset(self.dataset_args["train_file"]),
            "validation": self.build_dataset(self.dataset_args["validation_file"]),
            "test": self.build_dataset(self.dataset_args["test_file"]),
        }

        logger.info(
            f"Training set size: {len(self.datasets['train'])} - Validation set size: {len(self.datasets['validation'])} - Test set size: {len(self.datasets['test'])}"  # type: ignore
        )

    def train_dataloader(self) -> DataLoader:
        """Creates the dataloader for the traning step.

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
        """Creates the dataloader for the validation step.

        Returns:
            pytorch dataloader.
        """
        return DataLoader(
            self.datasets["validation"],  # type: ignore
            batch_size=1,
            num_workers=self.dataset_args["num_dataloader_workers"],
            collate_fn=self.data_collator,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates the dataloader for the test step.

        Returns:
            pytorch dataloader.
        """
        return DataLoader(
            self.datasets["test"],  # type: ignore
            batch_size=1,
            num_workers=self.dataset_args["num_dataloader_workers"],
            collate_fn=self.data_collator,
        )

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments for the dataset to the parser.

        Args:
            parent_parser: argument parser.

        Returns:
            updated parser.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train_file", type=str, required=True)
        parser.add_argument("--validation_file", type=str, required=True)
        parser.add_argument("--test_file", type=str, required=True)
        parser.add_argument("--num_dataloader_workers", type=int, default=8)
        parser.add_argument("--max_length", type=int, default=278)
        parser.add_argument("--padding", type=str, default="max_length")
        parser.add_argument("--truncation", type=bool, default=True)
        parser.add_argument("--batch_size", type=int, default=32)

        return parser
