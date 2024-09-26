"""Pytorch Lightning implementation for approximated VanillaTransformer."""

from argparse import ArgumentParser
from typing import Dict, Union

from torchmetrics import MetricCollection

from ....approximation.controller import ModelApproximationController
from ....metrics.text_sequence_matching import SequenceMatchingMetric
from ....models.vanilla_transformer.model import VanillaTransformer
from ....utils.logging import setup_logger
from ..language_model_for_generation import LitApproximatedLanguageModelForGeneration

logger = setup_logger(__name__, logging_level="info")


class LitApproximatedVanillaTransformer(LitApproximatedLanguageModelForGeneration):
    """Pytorch lightning model for the approximated VanillaTransformer."""

    def __init__(
        self,
        model: VanillaTransformer,
        controller: ModelApproximationController,
        model_args: Dict[str, Union[float, int, str]],
        metrics: MetricCollection = MetricCollection([SequenceMatchingMetric()]),
        **kwargs,
    ) -> None:
        """Construct an LM lightning module.

        Args:
            model_args: model's arguments.
        """

        if not isinstance(model, VanillaTransformer):
            raise TypeError(
                f"The model you are trying to approximate is not of class '{VanillaTransformer}'. Build a specific PyTorch lightning model inheriting from 'LitApproximatedModel'."
            )

        super().__init__(
            model=model,
            controller=controller,
            model_args=model_args,
            metrics=metrics,
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
            LitApproximatedLanguageModelForGeneration.add_model_specific_args(
                parent_parser=parent_parser
            )
        )
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group("model_config_args")

        # model configuration arguments
        group.add_argument("--embedding_dim", type=int, default=256)
        group.add_argument("--ffnn_hidden_dim", type=int, default=2048)
        group.add_argument("--dropout", type=float, default=0.1)
        group.add_argument("--activation", type=str, default="relu")
        group.add_argument("--num_attention_heads", type=int, default=8)
        group.add_argument("--num_encoder_layers", type=int, default=4)
        group.add_argument("--num_decoder_layers", type=int, default=4)
        group.add_argument("--attention_mask_value", type=float, default=float("-inf"))
        group.add_argument("--init_std", type=float, default=0.02)
        group.add_argument("--max_position_embeddings", type=int, default=5000)
        group.add_argument("--max_length", type=int, default=278)
        group.add_argument("--num_beams", type=int, default=1)

        return parser
