"""Vanilla transformer configuration."""

from transformers.configuration_utils import PretrainedConfig


class VanillaTransformerConfig(PretrainedConfig):
    """VanillaTransformerConfig implementation."""

    model_type = "vanilla_transformer"

    def __init__(
        self,
        vocabulary_size: int = 569,
        embedding_dim: int = 256,
        ffnn_hidden_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        num_attention_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        attention_mask_value: float = float("-inf"),
        init_std: float = 0.02,
        max_position_embeddings: int = 5000,
        pad_token_id: int = 0,
        bos_token_id: int = 12,
        eos_token_id: int = 13,
        decoder_start_token_id: int = 12,
        is_encoder_decoder: bool = True,
        return_dict_in_generate: bool = False,
        num_beams: int = 1,
        max_length: int = 278,
        device: str = "cpu",
        **kwargs
    ) -> None:
        """Initialization of the configuation.

        Args:
            vocabulary_size: number of different tokens that can be represented. Defaults to 569.
            embedding_dim: dimensionality of the layers and the pooler layer. Defaults to 256.
            ffnn_hidden_dim: Dimensionality of the feed-forward layer in language modeling head. Defaults to 2048.
            dropout: the dropout probability for all fully connected layers in the embeddings, encoder, and pooler. Defaults to 0.1.
            activation: activation function to be used in the encoder and decoder layers. Defaults to "relu".
            num_attention_heads: number of attention heads for each attention layer in the Transformer encoder and decoder. Defaults to 8.
            num_encoder_layers: number of encoder layers. Defaults to 4.
            num_decoder_layers: number of decoder layers. Defaults to 4.
            attention_mask_value: value used to mask attention values. Defaults to -inf.
            init_std: standard deviation of the truncated_normal_initializer for initializing all weight matrices. Defaults to 0.02.
            max_position_embeddings: maximum length used to compute the positional encoding. Defaults to 5000.
            pad_token_id: token value reserved for padding. Defaults to 0.
            bos_token_id: token value reserved for the beginning of a sentence. Defaults to 2.
            eos_token_id: token value reserved for the end of a sentence. Defaults to 3.
            decoder_start_token_id: token value given to the decoder as a start point for the generation. Defaults to 2.
            is_encoder_decoder: defines the structure of the model for the generation routine. Defaults to True.
            return_dict_in_generate: whether or not to return a [`ModelOutput`] instead of a plain tuple during generation. Defaults to False.
            num_beams: number of beams for beam search that will be used by default in the `generate` method of the model (1 means no beam search). Default to 1.
            max_length: maximum length of the sequence to be generated. Defaults to 278.
            device: device on which the model will be allocated. Defaults to "cpu".
        """

        # model hyperparameters configuration
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.ffnn_hidden_dim = ffnn_hidden_dim
        self.dropout = dropout
        self.activation = activation
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.attention_mask_value = attention_mask_value
        self.init_std = init_std
        self.max_position_embeddings = max_position_embeddings
        self.device = device

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            return_dict_in_generate=return_dict_in_generate,
            decoder_start_token_id=decoder_start_token_id,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs,
        )
