"""Vanilla transformer model."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import importlib_metadata  # type: ignore
import torch
import torch.utils.checkpoint
from packaging import version  # type: ignore
from torch import Tensor, nn
from torch.nn import (
    CrossEntropyLoss,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from transformers.modeling_utils import PreTrainedModel

from .configuration import VanillaTransformerConfig

# checking torch version
torch_version = version.parse(importlib_metadata.version("torch"))
TORCH_LT_1_9 = torch_version < version.parse("1.9")

# checking transformers version
transformers_version = version.parse(importlib_metadata.version("transformers"))
TRANSFORMERS_LT_4_12 = transformers_version < version.parse("4.12")
if TRANSFORMERS_LT_4_12:
    from transformers.file_utils import ModelOutput
else:
    from transformers.utils import ModelOutput


class TokenEmbedding(nn.Module):
    """Token embedding layer implementation."""

    def __init__(
        self, vocabulary_size: int, embedding_dim: int, padding_idx: int
    ) -> None:
        """Contructs a TokenEmbedding layer.

        Args:
            vocabulary_size: number of different tokens that can be represented.
            embedding_dim: dimensionality of the embeddings.
            padding_idx: token value reserved for padding.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx)
        self.embedding_dim = embedding_dim

    def forward(self, tokens: Tensor) -> Tensor:
        """Creation of the embedding.

        Args:
            tokens: tokens to be embedded.

        Returns:
            embedding of the input tokens.
        """
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    """Position embedding layer implementation."""

    def __init__(
        self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000
    ) -> None:
        """Initializes a PositionalEncoding layer.

        Args:
            embedding_dim: dimensionality of the embeddings.
            dropout: dropout probability. Defaults to 0.1.
            max_len: upperbound to the length of the sequence for the generation of position. Defaults to 5000.
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).reshape(max_len, 1)
        div_term = torch.exp(
            -torch.arange(0, embedding_dim, 2) * math.log(10000) / embedding_dim
        )
        position_embedding = torch.zeros(max_len, embedding_dim)
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        position_embedding = (
            position_embedding.unsqueeze(-2)
            if TORCH_LT_1_9
            else position_embedding.unsqueeze(0)
        )

        self.register_buffer("position_embedding", position_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        """Creation of the positional embedding.

        Args:
            token_embedding: token embedding to be merged to the positional embedding.

        Returns:
            Embedding obtained as the sum of token and positional embeddings.
        """
        if TORCH_LT_1_9:
            position_embedding = getattr(self, "position_embedding")[
                : token_embedding.size(0), :
            ]
        else:
            position_embedding = getattr(self, "position_embedding")[
                :, : token_embedding.size(1), :
            ]
        token_embedding = token_embedding + position_embedding
        return self.dropout(token_embedding)


class ExtendedTransformerEncoderLayer(TransformerEncoderLayer):
    """TransformerEncoderLayer extension to be compliant with different torch versions."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        """Initializes a TrasformerEncoderLayer.

        Args:
            d_model: the number of expected features in the input.
            nhead: the number of heads in the multiheadattention models.
            dim_feedforward: the dimension of the feedforward network model. Defaults to 2048.
            dropout: the dropout value. Defaults to 0.1.
            activation: the activation function of the intermediate layer. Defaults to "relu".
        """

        if TORCH_LT_1_9:
            super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        else:
            super().__init__(  # type: ignore
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                batch_first=kwargs.get("batch_first", False),
                device=kwargs.get("device", None),
            )


class ExtendedTransformerDecoderLayer(TransformerDecoderLayer):
    """TransformerEncoderLayer extension to be compliant with different torch versions."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        """Initializes a TrasformerDecoderLayer.

        Args:
            d_model: the number of expected features in the input.
            nhead: the number of heads in the multiheadattention models.
            dim_feedforward: the dimension of the feedforward network model. Defaults to 2048.
            dropout: the dropout value. Defaults to 0.1.
            activation: the activation function of the intermediate layer. Defaults to "relu".
        """

        if TORCH_LT_1_9:
            super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        else:
            super().__init__(  # type: ignore
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                batch_first=kwargs.get("batch_first", False),
                device=kwargs.get("device", None),
            )


class ExtendedLayerNorm(nn.LayerNorm):
    """LayerNorm extension to be compliant with different torch versions."""

    def __init__(
        self,
        normalized_shape: Union[int, List, torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        **kwargs,
    ):
        """Initializes a LayerNorm.

        Args:
            normalized_shape: input shape from an expected input of size.
            eps: a value added to the denominator for numerical stability. Defaults to 1e-5.
            elementwise_affine: a boolean value that when set to ``True``, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases). Defaults to True.
        """

        if TORCH_LT_1_9:
            super().__init__(normalized_shape, eps, elementwise_affine)
        else:
            super().__init__(  # type: ignore
                normalized_shape,
                eps,
                elementwise_affine,
                device=kwargs.get("device", None),
            )


class VanillaTransformerPretrainedModel(PreTrainedModel):
    """Abstract class to handle weights initialization."""

    config_class = VanillaTransformerConfig
    base_model_prefix = "model"

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes the weights.

        Args:
            module: layer of the model.
        """

        if isinstance(module, nn.Linear):
            # initilaization of the weights of a linear module
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # initialization of the weights of an embedding module
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # initialization of the weights of a layer normalization module
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class VanillaEncoderOutput(ModelOutput):
    """VanillaEncoderOutput implementation. Represents the output of the encoder block."""

    last_hidden_state: torch.FloatTensor


class VanillaTransformerEncoder(VanillaTransformerPretrainedModel):
    """VanillaTransformerEncoder implementation."""

    def __init__(self, config: VanillaTransformerConfig) -> None:
        """Initializes a VanillaTransformerEncoder.

        Args:
            config: configuration of the VanillaTransformer.
        """
        super().__init__(config)

        # initialize of the arguments to be passed or not to the extended classes depending on torch version
        kwargs = (
            {}
            if TORCH_LT_1_9
            else {
                "batch_first": True,
                "device": config.device,
            }
        )

        # set how the data is expected by torch encoder and decoder modules
        self.batch_first = kwargs.get("batch_first", False)

        # instanciating the embedding layer for the input tokens
        self.src_token_embedding = TokenEmbedding(
            vocabulary_size=config.vocabulary_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id,
        )
        # instanciating the embedding layer for the position of input tokens
        self.positional_encoding = PositionalEncoding(
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            max_len=config.max_position_embeddings,
        )
        # instanciating an encoder layer
        encoder_layer = ExtendedTransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.ffnn_hidden_dim,
            dropout=config.dropout,
            activation=config.activation,
            **kwargs,
        )
        # instanciating the normalization layer of the encoder
        encoder_norm = ExtendedLayerNorm(
            normalized_shape=config.embedding_dim,
            **kwargs,  # type: ignore
        )
        # instanciating the encoder block using `config.num_encoder_layers` encoder layers
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=encoder_norm,
        )

        # initialization of the weights
        if TRANSFORMERS_LT_4_12:
            self.init_weights()
        else:
            self.post_init()

    def _make_encoder_masks(
        self,
        input: Tensor,
        mask_size: int,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Create the attention masks for the encoder.

        Args:
            input: input of the encoder.
            mask_size: shape of the input.
            padding_mask: mask for the padding tokens. Defaults to None.

        Returns:
            tuple containing:
            - mask for the encoder input sequence.
            - padding mask for the encoder input sequence.
        """

        mask = torch.zeros(
            (mask_size, mask_size),
            device=self.config.device,
        ).type(torch.bool)

        if padding_mask is None:
            # create the padding mask if is not available
            padding_mask = input.eq(self.config.pad_token_id)
            # the padding mask is always taken with batch_first
            padding_mask = (
                padding_mask if self.batch_first else padding_mask.transpose(1, 0)  # type: ignore
            )
        else:
            # adjust the dimensions of the padding mask in case it is available
            padding_mask = padding_mask.view(-1, padding_mask.shape[-1])
            # the padding mask is always taken with batch_first

        return mask, padding_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> VanillaEncoderOutput:
        """Implements a forward pass through the VanillaDecoder.

        Args:
            input_ids: tokens given as input to the encoder.

        Raises:
            ValueError: in case of missing input_ids.

        Returns:
            a VanillaEncoderOutput.
        """

        # adjust input_ids dimension to allow not batched input i.e. of size (seq_length,)
        if input_ids is not None:
            input = input_ids.view(-1, input_ids.shape[-1])
            input_shape = input.shape
        else:
            raise ValueError("You have to specify input_ids.")

        # permute the dimension of the input_ids in case it is not batch_first
        input = input if self.batch_first else input.permute(1, 0)

        # compute input embeddings
        inputs_embeds = self.positional_encoding(self.src_token_embedding(input))

        # create masks required by the encoder
        src_mask, src_padding_mask = self._make_encoder_masks(
            input=input, mask_size=input_shape[1], padding_mask=padding_mask
        )

        # compute encoder output
        encoder_output = self.encoder(
            inputs_embeds,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask,
        )

        # permute the dimension to batch first in case the output is computed without batch_first
        encoder_output = (
            encoder_output if self.batch_first else encoder_output.permute(1, 0, 2)
        )

        return VanillaEncoderOutput(last_hidden_state=encoder_output)


@dataclass
class VanillaDecoderOutput(ModelOutput):
    """VanillaDecoderOutput implementation. Represents the output of the decoder block."""

    last_hidden_state: torch.FloatTensor


class VanillaTransformerDecoder(VanillaTransformerPretrainedModel):
    """VanillaTransformerDecoder implementation."""

    def __init__(self, config: VanillaTransformerConfig) -> None:
        """Initializes a VanillaTransformerDecoder.

        Args:
            config: configuration of the VanillaTransformer.
        """
        super().__init__(config)

        # initialize of the arguments to be passed or not to the extended classes depending on torch version
        kwargs = (
            {}
            if TORCH_LT_1_9
            else {
                "batch_first": True,
                "device": config.device,
            }
        )

        # set how the data is expected by torch encoder and decoder modules
        self.batch_first = kwargs.get("batch_first", False)

        # instanciating the embedding layer for the input tokens
        self.tgt_token_embedding = TokenEmbedding(
            config.vocabulary_size, config.embedding_dim, config.pad_token_id
        )

        # instanciating the embedding layer for the position of input tokens
        self.positional_encoding = PositionalEncoding(
            config.embedding_dim,
            dropout=config.dropout,
            max_len=config.max_position_embeddings,
        )

        # instanciating an decoder layer
        decoder_layer = ExtendedTransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.ffnn_hidden_dim,
            dropout=config.dropout,
            activation=config.activation,
            **kwargs,
        )

        # instanciating the normalization layer of the decoder
        decoder_norm = ExtendedLayerNorm(
            normalized_shape=config.embedding_dim,
            **kwargs,  # type: ignore
        )

        # instanciating the decoder block using `config.num_encoder_layers` decoder layers
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.num_decoder_layers,
            norm=decoder_norm,
        )

        # initializing the weights
        if TRANSFORMERS_LT_4_12:
            self.init_weights()
        else:
            self.post_init()

    def _make_square_subsequent_mask(
        self, mask_size: int, device: torch.device
    ) -> Tensor:
        """Creates the Look-ahead mask, needed to mask future inputs for the decoder.

        Args:
            mask_size: size of the mask.
            device: device on which the mask tensor will be allocated.

        Returns:
            the look-ahead mask.
        """
        return torch.triu(
            torch.ones(mask_size, mask_size) * self.config.attention_mask_value,
            diagonal=1,
        ).to(device)

    def _make_decoder_masks(
        self,
        input: Tensor,
        mask_size: int,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Create the masks for the decoder.

        Args:
            input: input of the decoder.
            mask_size: shape of the input.
            padding_mask: mask for the padding tokens. Defaults to None.

        Returns:
            tuple containing:
            - mask for the decoder input sequence.
            - padding mask for the decoder input sequence.
        """

        # creating the look-ahead mask
        mask = self._make_square_subsequent_mask(mask_size, self.config.device)

        if padding_mask is None:
            # creating the padding mask if is not available
            padding_mask = input.eq(self.config.pad_token_id)
            # the padding mask is always taken with batch_first
            padding_mask = (
                padding_mask if self.batch_first else padding_mask.transpose(1, 0)  # type: ignore
            )
        else:
            # adjusting the dimensions padding mask in case it is available
            padding_mask = padding_mask.view(-1, padding_mask.shape[-1])
            # the padding mask is always taken with batch_first

        return mask, padding_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_output: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> VanillaDecoderOutput:
        """Implements a forward pass through the VanillaDecoder.

        Args:
            input_ids: tokens given as input to the decoder.
            encoder_output: hidden state created by the encoder. Defaults to None.

        Raises:
            ValueError: in case of missing input_ids.

        Returns:
            A VanillaDecoderOutput.
        """

        # adjusting input_ids dimension to allow not batched input i.e. of size (seq_length,)
        if input_ids is not None:
            input = input_ids.view(-1, input_ids.shape[-1])
            input_shape = input.shape
        else:
            raise ValueError("You have to specify decoder_input_ids.")

        # permuting the dimension of the input_ids in case it is not batch_first
        input = input if self.batch_first else input.permute(1, 0)

        # computing input embeddings
        inputs_embeds = self.positional_encoding(self.tgt_token_embedding(input))

        # creating masks required by the decoder
        tgt_mask, tgt_padding_mask = self._make_decoder_masks(
            input=input,
            mask_size=input_shape[1],
            padding_mask=padding_mask,
        )

        # computing decoder's output
        decoder_output = self.decoder(
            inputs_embeds,
            encoder_output,
            tgt_mask=tgt_mask,
            # memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            # memory_key_padding_mask=memory_key_padding_mask
        )

        # permuting the dimension to batch_first in case the output is computed without batch_first
        decoder_output = (
            decoder_output if self.batch_first else decoder_output.permute(1, 0, 2)
        )

        return VanillaDecoderOutput(
            last_hidden_state=decoder_output,
        )


@dataclass
class VanillaTransformerOutput(ModelOutput):
    """VanillaDecoderOutput implementation.

    Args:
        loss: language modeling loss. Must be the first argument for huggingface Trainer.
        logits: prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None  # type: ignore


class VanillaTransformer(VanillaTransformerPretrainedModel):
    """VanillaTransformer implementation."""

    def __init__(self, config: VanillaTransformerConfig) -> None:
        """Initializes a VanillaTransformer.

        Args:
            config: configuration of the VanillaTransformer.
        """
        super().__init__(config)

        # instanciating the vanilla transformer encoder
        self.encoder = VanillaTransformerEncoder(config)
        # instanciating the vanilla transformer decoder
        self.decoder = VanillaTransformerDecoder(config)
        # instanciating the prediction head
        self.lm_head = nn.Linear(config.embedding_dim, config.vocabulary_size)

        # defining the loss function
        self.loss_function = CrossEntropyLoss(ignore_index=config.pad_token_id)

        # initializing the weights (i.e. calling _init_weights of VanillaTransformerPretrainedModel)
        if TRANSFORMERS_LT_4_12:
            self.init_weights()
        else:
            self.post_init()

    def forward(
        self,
        encoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        decoder_padding_mask: Optional[Tensor] = None,
        encoder_output: Optional[ModelOutput] = None,
        **kwargs,
    ) -> VanillaTransformerOutput:
        """Implements a forward pass through the VanillaTransformer.

        Args:
            encoder_input_ids: tokens given as input to the encoder. Defaults to None.
            decoder_input_ids: tokens given as input to the decoder. Defaults to None.
            encoder_output: hidden state created by the encoder. Defaults to None.

        Raises:
            ValueError: in case of missing decoder_input_ids.
            ValueError: in case the encoder output is not computed.

        Returns:
            a VanillaTransformerOutput.
        """

        # adjusting encoder_input_ids dimension to allow not batched input i.e. of size (seq_length,)
        if encoder_input_ids is not None:
            input_shape = encoder_input_ids.size()
            input = encoder_input_ids.view(-1, input_shape[-1])

        # adjusting decoder_input_ids dimension to allow not batched input i.e. of size (seq_length,)
        if decoder_input_ids is not None:
            target_shape = decoder_input_ids.size()
            target = decoder_input_ids.view(-1, target_shape[-1])
        else:
            raise ValueError(
                "The value of decoder_input_ids must be specified. Is None"
            )

        # adjusting the dimensions of the decoder padding mask in case it is available
        if decoder_padding_mask is not None:
            decoder_padding_mask = decoder_padding_mask.view(
                -1, decoder_padding_mask.size()[-1]
            )

        # adjusting the dimensions of the encoder padding mask in case it is available
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.view(
                -1, encoder_padding_mask.size()[-1]
            )

        if encoder_input_ids is None:
            target_input = target
        else:
            target_input = target[:, :-1]  # remove the last element of the sequence
            if decoder_padding_mask is not None:
                decoder_padding_mask = decoder_padding_mask[:, :-1]

        # encoding the sequence
        if encoder_input_ids is not None and encoder_output is None:
            encoder_output = self.encoder(
                input_ids=input, padding_mask=encoder_padding_mask
            )

        # decoding the sequence
        if encoder_output is not None:
            decoder_output = self.decoder(
                input_ids=target_input,
                encoder_output=encoder_output.last_hidden_state,
                padding_mask=decoder_padding_mask,
            )
        else:
            raise ValueError("The value of encoder_output must be computed. Is None.")

        # predicting the next token
        logits = self.lm_head(decoder_output.last_hidden_state)

        # shifting the tgt by one so with the <SOS> we predict the token at pos 1
        target_output = target[:, 1:]

        if encoder_input_ids is not None:
            loss = self.loss_function(
                logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1)
            )
        else:
            loss = None

        return VanillaTransformerOutput(
            loss=loss,
            logits=logits,
        )

    def encode(
        self, input_ids: Tensor, padding_mask: Optional[Tensor] = None
    ) -> VanillaEncoderOutput:
        """Encodes the input through a forward pass in the encoder of the model.

        Needed for the generation of new tokens.

        Args:
            input_ids: tokens given as input to the encoder.
            padding_mask: mask for the padding tokens in the input. Defaults to None.

        Returns:
            the encoder output.
        """
        return self.encoder(input_ids=input_ids, padding_mask=padding_mask)

    def decode(
        self,
        input_ids: Tensor,
        encoder_output: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode the target through a forward pass in the decoder of the model.

        Needed for the generation of new tokens.

        Args:
            input_ids: tokens given as input to the decoder.
            encoder_output: output of the encoder.
            padding_mask: mask for the padding tokens in the input. Defaults to None.

        Returns:
            the decoder output.
        """
        return self.decoder(
            input_ids=input_ids,
            encoder_output=encoder_output,
            padding_mask=padding_mask,
        )

    def prepare_inputs_for_generation(
        self, decoder_input_ids: Tensor, **kwargs
    ) -> Dict[str, Any]:
        """Prepare the input for the foward pass of the model during generation.

        It is called during the generation, passing the input for the decoder and the encoded input sequence.

        Args:
            decoder_input_ids: tokens given as input to the decoder.

        Returns:
            dictionary with the arguments for a forward pass through the vanilla transformer.
        """

        return {
            "encoder_input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_output": kwargs.get("encoder_outputs"),
            "decoder_input_ids": decoder_input_ids,
        }

    def get_encoder(self):
        """Returns the encoder of the model.

        Needed for the generation of new tokens.

        Returns:
            encoder module of the vanilla transformer.
        """
        return self.encoder

    def get_decoder(self):
        """Returns the decoder of the model.

        Needed for the generation of new tokens.

        Returns:
            decoder module of the vanilla transformer.
        """
        return self.decoder
