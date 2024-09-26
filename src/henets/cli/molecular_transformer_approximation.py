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
    importlib_resources.files("henets")
    / "resources"
    / "models"
    / "tokenizers"
    / "smiles_vocab.txt"
)


def main():
    pl.seed_everything(42)

    trainer_parser = ArgumentParser()
    # adding to the parser the arguments needed from each component
    trainer_parser = pl.Trainer.add_argparse_args(trainer_parser)  # type: ignore
    trainer_parser = ApproximationPipeline.add_pipeline_specific_args(trainer_parser)
    trainer_args, remaining_args = trainer_parser.parse_known_args()

    model_config_parser = ArgumentParser()
    model_config_parser = LitApproximatedVanillaTransformer.add_model_specific_args(
        model_config_parser
    )
    model_config_args, remaining_args = model_config_parser.parse_known_args(
        remaining_args
    )

    dataset_parser = ArgumentParser()
    dataset_parser = LitSmilesDataset.add_dataset_specific_args(dataset_parser)
    dataset_parser.add_argument("--vocabulary", type=str, default=SMILES_VOCAB_FILE)
    dataset_args, remaining_args = dataset_parser.parse_known_args(remaining_args)

    # building the lightning dataset
    tokenizer = SmilesTokenizer(vars(dataset_args)["vocabulary"])
    smiles_dataset = LitSmilesDataset(
        dataset_args=vars(dataset_args), tokenizer=tokenizer
    )

    smiles_dataset.load()
    train_dataloader = smiles_dataset.train_dataloader()
    val_dataloader = smiles_dataset.val_dataloader()
    test_dataloader = smiles_dataset.test_dataloader()

    model_config_dict: dict = {}
    for group in model_config_parser._action_groups:
        group_args = [action.dest for action in group._group_actions]
        group_values = {
            arg: getattr(model_config_args, arg, None)
            for arg in group_args
            if hasattr(model_config_args, arg)
        }
        model_config_dict[group.title] = group_values

    config_args = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.sep_token_id,
        "bos_token_id": tokenizer.cls_token_id,
        "decoder_start_token_id": tokenizer.cls_token_id,
        "vocabulary_size": tokenizer.vocab_size,
        "device": (
            "cpu"
            if vars(trainer_args)["accelerator"] == "cpu"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
    }

    # building the pytorch model
    model = VanillaTransformer(
        VanillaTransformerConfig(
            **model_config_dict["model_config_args"], **config_args
        )
    )

    pipeline = ApproximationPipeline(
        model=model,
        lightning_model_class=LitApproximatedVanillaTransformer,
        lightning_model_args=vars(model_config_args),
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        trainer_args=vars(trainer_args),
        pipeline_steps_path=vars(trainer_args)["pipeline_steps_path"],
        modules_aliases_file=vars(trainer_args)["modules_aliases_file"],
        experiment_ckpt=vars(trainer_args)["experiment_ckpt"],
    )

    # training and validating the model through the pipeline
    pipeline.fit()

    # testing the model obtained through the pipeline
    pipeline.test()
