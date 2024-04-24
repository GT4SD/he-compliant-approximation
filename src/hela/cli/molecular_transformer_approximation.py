"""Routine to train and test an approximated Molecular Transformer model using a pipeline."""

from argparse import ArgumentParser

import importlib_resources
import pytorch_lightning as pl
import torch

from ..approximation.pipeline.approximation_pipeline import ApproximationPipeline
from ..models.tokenizers.smiles import SmilesTokenizer
from ..models.vanilla_transformer.configuration import VanillaTransformerConfig
from ..models.vanilla_transformer.model import VanillaTransformer
from ..pytorch_lightning.datasets.smiles import LitSmilesDataset
from ..pytorch_lightning.models.language.vanilla_transformer import (
    LitApproximatedVanillaTransformer,
)

SMILES_VOCAB_FILE = str(
    importlib_resources.files("hela")
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
    parser = LitApproximatedVanillaTransformer.add_model_specific_args(parser)
    parser = LitSmilesDataset.add_dataset_specific_args(parser)
    parser.add_argument("--vocabulary", type=str, default=SMILES_VOCAB_FILE)
    parser = ApproximationPipeline.add_pipeline_specific_args(parser)
    args = parser.parse_args()

    # building the lightning dataset
    tokenizer = SmilesTokenizer(vars(args)["vocabulary"])
    smiles_dataset = LitSmilesDataset(dataset_args=vars(args), tokenizer=tokenizer)

    smiles_dataset.load()
    train_dataloader = smiles_dataset.train_dataloader()
    val_dataloader = smiles_dataset.val_dataloader()
    test_dataloader = smiles_dataset.test_dataloader()

    config_args = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.sep_token_id,
        "bos_token_id": tokenizer.cls_token_id,
        "decoder_start_token_id": tokenizer.cls_token_id,
        "vocabulary_size": tokenizer.vocab_size,
        "num_encoder_layers": vars(args)["num_encoder_layers"],
        "num_decoder_layers": vars(args)["num_decoder_layers"],
        "attention_mask_value": vars(args)["attention_mask_value"],
        "device": (
            "cpu"
            if vars(args)["accelerator"] == "cpu"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
    }

    # building the pytorch model
    model = VanillaTransformer(VanillaTransformerConfig(**config_args))

    pipeline = ApproximationPipeline(
        model=model,
        lightning_model_class=LitApproximatedVanillaTransformer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        trainer_args=vars(args),
        pipeline_steps_path=vars(args)["pipeline_steps_path"],
        modules_aliases_file=vars(args)["modules_aliases_file"],
        experiment_ckpt=vars(args)["experiment_ckpt"],
    )

    # training and validating the model through the pipeline
    pipeline.fit()

    # testing the model obtained through the pipeline
    pipeline.test()
