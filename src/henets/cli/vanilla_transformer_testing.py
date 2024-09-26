"""Routine to train an approximated model using a pipeline."""

from argparse import ArgumentParser

import importlib_resources
import pytorch_lightning as pl

from ..models.tokenizers.smiles import SmilesTokenizer
from ..pytorch_lightning.datasets.smiles import LitSmilesDataset
from ..pytorch_lightning.models.vanilla_transformer.model import LitVanillaTransformer

SMILES_VOCAB_FILE = str(
    importlib_resources.files("henets")
    / "resources"
    / "models"
    / "tokenizers"
    / "smiles_vocab.txt"
)


def main():
    pl.seed_everything(42)

    parser = ArgumentParser()
    # adding to the parser the arguments needed from each component
    parser = pl.Trainer.add_argparse_args(parser)  # type: ignore
    parser = LitVanillaTransformer.add_model_specific_args(parser)
    parser = LitSmilesDataset.add_dataset_specific_args(parser)
    parser.add_argument("--vocabulary", type=str, default=SMILES_VOCAB_FILE)
    parser.add_argument("--ckpt", type=str, default=None, required=True)
    args = parser.parse_args()

    # building the lightning dataset
    tokenizer = SmilesTokenizer(vars(args)["vocabulary"])
    smiles_dataset = LitSmilesDataset(dataset_args=vars(args), tokenizer=tokenizer)

    smiles_dataset.load()
    test_dataloader = smiles_dataset.test_dataloader()

    # defining the lightning model
    model = LitVanillaTransformer(model_args=vars(args), tokenizer=tokenizer)

    # testing the model loading the model from the specified checkpoint
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.test(model, dataloaders=test_dataloader, ckpt_path=vars(args)["ckpt"])
