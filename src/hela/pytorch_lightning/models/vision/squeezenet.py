"""Pytorch Lightning implementation for approximated SqueezeNet."""

from argparse import ArgumentParser
from typing import Dict, Union

from ....approximation.controller import ModelApproximationController
from ....models.squeezenet.model import SqueezeNet
from ....utils.logging import setup_logger
from ..vision_model_for_classification import (
    LitApproximatedVisionModelForClassification,
)

logger = setup_logger(__name__, logging_level="info")


class LitApproximatedSqueezeNet(LitApproximatedVisionModelForClassification):
    """Pytorch lightning model for the approximated SqueezeNet."""

    def __init__(
        self,
        model: SqueezeNet,
        controller: ModelApproximationController,
        model_args: Dict[str, Union[float, int, str]],
    ) -> None:
        """Construct a SqueezeNet lightning module.

        Args:
            model_args: model's arguments.
        """

        if not isinstance(model, SqueezeNet):
            raise TypeError(
                f"The model you are trying to approximate is not of class '{SqueezeNet}'. Build a specific PyTorch lightning model inheriting from 'LitApproximatedModel'."
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
        parent_parser = (
            LitApproximatedVisionModelForClassification.add_model_specific_args(
                parent_parser=parent_parser
            )
        )
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group("model_config_args")

        # model configuration arguments
        group.add_argument("--model_version", type=str, default="1_0")
        group.add_argument("--dropout", type=float, default=0.5)

        return parser
