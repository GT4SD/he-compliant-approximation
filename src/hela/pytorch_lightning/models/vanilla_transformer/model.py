"""Pytorch Lightning implementation for VanillaTransformer."""

import logging
from argparse import ArgumentParser
from typing import Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import torch.optim as optim
from ....models.tokenizers.smiles import SmilesTokenizer
from torch import Tensor

from ....models.vanilla_transformer.configuration import VanillaTransformerConfig
from ....models.vanilla_transformer.model import VanillaTransformer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LitVanillaTransformer(pl.LightningModule):
    """Pytorch lightning model for VanillaTransformer."""

    def __init__(
        self,
        model_args: Dict[str, Union[float, int, str]],
        tokenizer: SmilesTokenizer,
    ) -> None:
        """Construct an LM lightning module.

        Args:
            model_args: model's arguments.
        """
        super().__init__()

        self.save_hyperparameters()

        self.model_args = model_args

        self.model: VanillaTransformer
        self.tokenizer = tokenizer

        self.init_model()

    def init_model(self) -> None:
        """Initialize a VanillaTransformer."""

        config_args = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.sep_token_id,
            "bos_token_id": self.tokenizer.cls_token_id,
            "decoder_start_token_id": self.tokenizer.cls_token_id,
            "vocabulary_size": self.tokenizer.vocab_size,
            "embedding_dim": self.model_args["embedding_dim"],
            "ffnn_hidden_dim": self.model_args["ffnn_hidden_dim"],
            "dropout": self.model_args["dropout"],
            "activation": self.model_args["activation"],
            "num_attention_heads": self.model_args["num_attention_heads"],
            "num_encoder_layers": self.model_args["num_encoder_layers"],
            "num_decoder_layers": self.model_args["num_decoder_layers"],
            "init_std": self.model_args["init_std"],
            "max_position_embeddings": self.model_args["max_position_embeddings"],
            "num_beams": self.model_args["num_beams"],
        }

        if self.model_args["model_name_or_path"] is not None:
            self.model = VanillaTransformer.from_pretrained(
                self.model_args["model_name_or_path"],
            )
            logger.info(
                f"Model from pretrained: {self.model_args['model_name_or_path']}."
            )
        else:
            if self.model_args["model_config_name"] is not None:
                config = VanillaTransformerConfig.from_pretrained(
                    self.model_args["model_config_name"]
                )
                logger.info(
                    f"Configuration from pretrained: {self.model_args['model_config_name']}."
                )
            else:
                config = VanillaTransformerConfig(**config_args)
                logger.info("Default configuration.")

            self.model = VanillaTransformer(config)

            logger.info("Training from scratch")

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        """Forwards through the model.

        Raises:
            NotImplementedError: implement this method for the prediction step.
        """
        raise NotImplementedError(
            "Implement the forward method for the LitVanillaTransformer."
        )

    def configure_optimizers(
        self,
    ) -> Dict[str, object]:
        """Create and return the optimizer.

        Returns:
            output (dict of str: Any):
                - optimizer: the optimizer used to update the parameter.
        """

        if not isinstance(self.model_args["learning_rate"], float):
            raise ValueError("Learning rate should be float")

        # definition of the optimizer
        optimizer = optim.AdamW(
            params=self.parameters(),
            lr=self.model_args["learning_rate"],  # type: ignore
            betas=(self.model_args["adam_beta1"], self.model_args["adam_beta2"]),  # type: ignore
            eps=self.model_args["adam_epsilon"],  # type: ignore
            weight_decay=self.model_args["adam_weight_decay"],  # type: ignore
        )

        output = {
            "optimizer": optimizer,
        }

        return output  # type: ignore

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        """
        Training step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            loss computed on the batch.
        """
        loss = self.model(**batch).loss  # type:ignore
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        """
        Validation step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            loss computed on the batch.
        """
        loss = self.model(**batch).loss  # type:ignore
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> float:  # type: ignore
        """
        Test step which encompasses the forward pass and the computation of the accuracy and the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            accuracy computed on the batch.
        """

        input_ids = batch["encoder_input_ids"]
        labels = batch["decoder_input_ids"]

        # generating the predicted sequence
        predictions = self.model.generate(
            input_ids,
            do_sample=False,
            max_length=self.model_args["max_length"],
            num_beams=self.model_args["num_beams"],
        )

        # padding the predicted sequence
        predictions = F.pad(
            predictions,
            pad=(0, self.model_args["max_length"] - predictions.shape[1]),  # type: ignore
            value=self.model.config.pad_token_id,
        )

        # check if the prediction perfectly matches the labels
        accuracy = 1.0 if torch.equal(predictions.squeeze(), labels.squeeze()) else 0.0

        self.log("accuracy", accuracy, prog_bar=True)

        return accuracy

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds model specific arguments to the parser.

        Args:
            parent_parser: argument parser.

        Returns:
            updated parser.
        """

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # training configuration arguments
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.98)
        parser.add_argument("--adam_epsilon", type=float, default=1e-9)
        parser.add_argument("--adam_weight_decay", type=float, default=0.01)
        parser.add_argument("--model_name_or_path", type=str, default=None)
        parser.add_argument("--model_config_name", type=str, default=None)

        # model configuration arguments
        parser.add_argument("--embedding_dim", type=int, default=256)
        parser.add_argument("--ffnn_hidden_dim", type=int, default=2048)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--num_attention_heads", type=int, default=8)
        parser.add_argument("--num_encoder_layers", type=int, default=4)
        parser.add_argument("--num_decoder_layers", type=int, default=4)
        parser.add_argument("--attention_mask_value", type=float, default=float("-inf"))
        parser.add_argument("--init_std", type=float, default=0.02)
        parser.add_argument("--max_position_embeddings", type=int, default=5000)
        parser.add_argument("--num_beams", type=int, default=1)

        return parser
