"""Pytorch Lightning implementation for approximated LeNet."""

import logging
from argparse import ArgumentParser
from typing import Dict, Union

from ....approximation.controller import ModelApproximationController
from ....models.lenet.model import LeNet
from .core import LitApproximatedCNN

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LitApproximatedLeNet(LitApproximatedCNN):
    """Pytorch lightning model for the approximated LeNet."""

    def __init__(
        self,
        model: LeNet,
        controller: ModelApproximationController,
        model_args: Dict[str, Union[float, int, str]],
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
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds model specific arguments to the parser.

        Args:
            parent_parser: argument parser.

        Returns:
            updated parser.
        """
        parent_parser = LitApproximatedCNN.add_model_specific_args(
            parent_parser=parent_parser
        )
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model configuration arguments
        parser.add_argument("--lenet_type", type=str, default="LeNet-5")
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--greyscale", dest="greyscale", action="store_true", default=False)

        return parser
