"""Routine to test an approximated LeNet model using a pipeline."""

from argparse import ArgumentParser

import pytorch_lightning as pl

from ..approximation.pipeline.testing import TestingPipeline
from ..models.lenet.configuration import LeNetConfig
from ..models.lenet.model import LeNet
from ..pytorch_lightning.datasets.torchvision_image_classification import (
    LitImageClassificationDataset,
)
from ..pytorch_lightning.models.approximations.lenet import LitApproximatedLeNet


def main():
    pl.seed_everything(42)

    parser = ArgumentParser()
    # adding to the parser the arguments needed from each component
    parser = pl.Trainer.add_argparse_args(parser)  # type: ignore
    parser = LitApproximatedLeNet.add_model_specific_args(parser)
    parser = LitImageClassificationDataset.add_dataset_specific_args(parser)
    parser = TestingPipeline.add_pipeline_specific_args(parser)
    args = parser.parse_args()

    # building the lightning dataset
    mnist_dataset = LitImageClassificationDataset(dataset_args=vars(args))

    mnist_dataset.load()
    test_dataloader = mnist_dataset.test_dataloader()

    config_args = {
        "lenet_type": vars(args)["lenet_type"],
        "num_classes": vars(args)["num_classes"],
        "greyscale": vars(args)["greyscale"],
    }

    # building the pytorch model
    model = LeNet(LeNetConfig(**config_args))

    pipeline = TestingPipeline(
        model=model,
        lightning_model_class=LitApproximatedLeNet,
        test_dataloader=test_dataloader,
        trainer_args=vars(args),
        pipeline_steps_path=vars(args)["pipeline_steps_path"],
        modules_aliases_file=vars(args)["modules_aliases_file"],
        experiment_ckpt=vars(args)["experiment_ckpt"],
    )

    # testing the model through the pipeline
    pipeline.test()
