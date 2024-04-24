"""Pytorch Lightning implementation for approximated LeNet."""

import logging
from argparse import ArgumentParser
from typing import Dict, Union

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from ....approximation.controller import ModelApproximationController
from ....models.lenet.model import LeNet
from ..vision_model_for_classification import (
    LitApproximatedVisionModelForClassification,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LitApproximatedLeNet(LitApproximatedVisionModelForClassification):
    """Pytorch lightning model for the approximated LeNet."""

    def __init__(
        self,
        model: LeNet,
        controller: ModelApproximationController,
        model_args: Dict[str, Union[float, int, str]],
        metrics=MetricCollection([MulticlassAccuracy(num_classes=10)]),
        **kwargs,
    ) -> None:
        """Construct a LeNet lightning module.

        Args:
            model_args: model's arguments.
        """

        if not isinstance(model, LeNet):
            raise TypeError(
                f"The model you are trying to approximate is not of class '{LeNet}'. Build a specific PyTorch lightning model inheriting from 'LitApproximatedModel'."
            )

        super().__init__(
            model=model,
            controller=controller,
            model_args=model_args,
            metrics=metrics,
            **kwargs,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds model specific arguments to the parser.

        Args:
            parent_parser: argument parser.

        Returns:
            updated parser.
        """
        parent_parser = (
            LitApproximatedVisionModelForClassification.add_model_specific_args(
                parent_parser=parent_parser
            )
        )
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group("model_config_args")

        # model configuration arguments
        group.add_argument("--lenet_type", type=str, default="lenet-5")

        return parser
