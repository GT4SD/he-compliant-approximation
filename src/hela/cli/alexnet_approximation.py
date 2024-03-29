"""Routine to train and test an approximated AlexNet model using a pipeline."""

from argparse import ArgumentParser

import pytorch_lightning as pl

from ..approximation.pipeline.approximation_pipeline import ApproximationPipeline
from ..models.alexnet.configuration import AlexNetConfig
from ..models.alexnet.model import AlexNet
from ..pytorch_lightning.datasets.torchvision_image_classification import (
    LitImageClassificationDataset,
)
from ..pytorch_lightning.models.approximations.alexnet import LitApproximatedAlexNet


def main():
    pl.seed_everything(42)

    parser = ArgumentParser()
    # adding to the parser the arguments needed from each component
    parser = pl.Trainer.add_argparse_args(parser)  # type: ignore
    parser = LitApproximatedAlexNet.add_model_specific_args(parser)
    parser = LitImageClassificationDataset.add_dataset_specific_args(parser)
    parser = ApproximationPipeline.add_pipeline_specific_args(parser)
    args = parser.parse_args()

    # building the lightning dataset
    dataset = LitImageClassificationDataset(dataset_args=vars(args))

    dataset.load()
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()
    test_dataloader = dataset.test_dataloader()

    config_args = {
        "num_classes": vars(args)["num_classes"],
        "dropout": vars(args)["dropout"],
    }

    # building the pytorch model
    model = AlexNet(AlexNetConfig(**config_args))

    pipeline = ApproximationPipeline(
        model=model,
        lightning_model_class=LitApproximatedAlexNet,
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
