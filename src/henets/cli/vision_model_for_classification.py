"""Routine to train and test an approximated vision models for image classification using a pipeline."""

from argparse import ArgumentParser

import pytorch_lightning as pl

from ..approximation.pipeline.approximation_pipeline import ApproximationPipeline
from ..models.alexnet.configuration import AlexNetConfig
from ..models.alexnet.model import AlexNet

# importing all the vision models
from ..models.lenet.configuration import LeNetConfig
from ..models.lenet.model import LeNet
from ..models.squeezenet.configuration import SqueezeNetConfig
from ..models.squeezenet.model import SqueezeNet
from ..pytorch_lightning.datasets.torchvision_image_classification import (
    LitImageClassificationDataset,
)
from ..pytorch_lightning.models.vision.alexnet import LitApproximatedAlexNet
from ..pytorch_lightning.models.vision.lenet import LitApproximatedLeNet
from ..pytorch_lightning.models.vision.squeezenet import (
    LitApproximatedSqueezeNet,
)

MODELS_AVAILABLE = {
    "lenet": {
        "config": LeNetConfig,
        "model": LeNet,
        "lightning_model_class": LitApproximatedLeNet,
    },
    "alexnet": {
        "config": AlexNetConfig,
        "model": AlexNet,
        "lightning_model_class": LitApproximatedAlexNet,
    },
    "squeezenet": {
        "config": SqueezeNetConfig,
        "model": SqueezeNet,
        "lightning_model_class": LitApproximatedSqueezeNet,
    },
}


def main():
    pl.seed_everything(42)

    model_parser = ArgumentParser()
    model_parser.add_argument("--model", type=str, required=True)
    model_args, remaining_args = model_parser.parse_known_args()
    assert (
        str(model_args.model).lower() in MODELS_AVAILABLE.keys()
    ), f"{model_args.model} not available, choose one of {MODELS_AVAILABLE.keys()}"
    model_dict = MODELS_AVAILABLE[str(model_args.model).lower()]

    # adding to the parser the arguments needed from each component
    trainer_parser = ArgumentParser()
    trainer_parser = pl.Trainer.add_argparse_args(trainer_parser)  # type: ignore
    trainer_parser = ApproximationPipeline.add_pipeline_specific_args(trainer_parser)
    trainer_args, remaining_args = trainer_parser.parse_known_args(remaining_args)

    model_config_parser = ArgumentParser()
    model_config_parser = model_dict["lightning_model_class"].add_model_specific_args(
        model_config_parser
    )
    model_config_args, remaining_args = model_config_parser.parse_known_args(
        remaining_args
    )

    dataset_parser = ArgumentParser()
    dataset_parser = LitImageClassificationDataset.add_dataset_specific_args(
        dataset_parser
    )
    dataset_args, remaining_args = dataset_parser.parse_known_args(remaining_args)

    if model_args.model == "lenet" and model_config_args.lenet_type == "lenet-1":
        dataset_args.image_size = 28

    # building the lightning dataset
    dataset = LitImageClassificationDataset(
        dataset_name=dataset_args.dataset_name,
        dataset_path=dataset_args.dataset_path,
        batch_size=dataset_args.batch_size,
        image_size=dataset_args.image_size,
        num_dataloader_workers=dataset_args.num_dataloader_workers,
        pin_memory=dataset_args.pin_memory,
        persistent_workers=dataset_args.persistent_workers,
    )

    dataset.prepare_data()
    dataset.setup()
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()
    test_dataloader = dataset.test_dataloader()

    num_classes = dataset.get_num_classes()
    greyscale = dataset.is_grayscale()

    # building the pytorch model
    model_config_dict: dict = {}

    for group in model_config_parser._action_groups:
        group_args = [action.dest for action in group._group_actions]
        group_values = {
            arg: getattr(model_config_args, arg, None)
            for arg in group_args
            if hasattr(model_config_args, arg)
        }
        model_config_dict[group.title] = group_values

    model = model_dict["model"](
        model_dict["config"](
            **model_config_dict["model_config_args"],
            num_classes=num_classes,
            greyscale=greyscale,
        )
    )

    pipeline = ApproximationPipeline(
        model=model,
        lightning_model_class=model_dict["lightning_model_class"],
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
