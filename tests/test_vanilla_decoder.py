"""Testing the vanilla transformer decoder."""

import os
from tempfile import TemporaryDirectory

import torch

from hela.models.vanilla_transformer.configuration import VanillaTransformerConfig
from hela.models.vanilla_transformer.model import (
    VanillaDecoderOutput,
    VanillaTransformerDecoder,
)

# default device to run the tests
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_decoder_init():
    """Tests the initialization of a vanilla transformer decoder."""

    # initializing the (default) vanilla transformer configuration
    configuration = VanillaTransformerConfig()

    # initializing a vanilla transformer decoder model from the configuration
    model = VanillaTransformerDecoder(configuration)

    # ASSERTS

    assert isinstance(model, VanillaTransformerDecoder)


def test_decoder_save_pretrained():
    """Tests the saving of a vanilla transformer decoder model and configuration."""

    with TemporaryDirectory() as temp_dir:
        # initializing a vanilla transformer decoder from the (default) configuration
        model = VanillaTransformerDecoder(VanillaTransformerConfig())

        # saving the randomly initialized model and the (default) configuration
        model.save_pretrained(save_directory=temp_dir)

        # ASSERTS

        assert isinstance(model, VanillaTransformerDecoder)
        assert os.path.isdir(temp_dir)
        assert sorted(os.listdir(temp_dir)) == sorted(
            ["config.json", "pytorch_model.bin"]
        )


def test_decoder_load_pretrained():
    """Tests the loading of a vanilla transformer decoder model and configuration."""

    with TemporaryDirectory() as temp_dir:
        # initializing a vanilla transformer decoder from the configuration
        decoder = VanillaTransformerDecoder(VanillaTransformerConfig())

        # saving the randomly initialized model and the (default) configuration
        decoder.save_pretrained(save_directory=temp_dir)

        # loading the saved model and configuration
        loaded_decoder = VanillaTransformerDecoder.from_pretrained(temp_dir)

        # ASSERTS

        assert isinstance(decoder, VanillaTransformerDecoder)
        assert isinstance(loaded_decoder, VanillaTransformerDecoder)

        for name, param in decoder.named_parameters():
            value = loaded_decoder
            for attr in name.split("."):
                value = getattr(value, attr)
            assert torch.equal(param, value)


def test_decoder_forward_pass_with_no_batched_input():
    """Tests the forward pass through a vanilla transformer decoder.
    The input is not batched.
    """

    # initializing a vanilla transformer decoder from the (default) configuration
    decoder = VanillaTransformerDecoder(VanillaTransformerConfig())
    # moving the decoder to the default DEVICE
    decoder.to(DEVICE)

    # creating a dummy input of size (sequence_length) and (1, sequence_length, embedding_dim)
    sequence_length = 10
    embedding_dimension = decoder.config.embedding_dim
    input_ids = torch.ones((sequence_length,), device=DEVICE).long()
    encoder_output = torch.ones(
        (1, sequence_length, embedding_dimension), device=DEVICE
    ).float()

    # forwarding through the decoder
    output = decoder(input_ids=input_ids, encoder_output=encoder_output)

    # ASSERTS

    assert isinstance(output, VanillaDecoderOutput)


def test_decoder_forward_pass_with_no_batched_input_and_padding_mask():
    """Tests the forward pass through a vanilla transformer decoder.
    The input is not batched and the decoder padding mask is given.
    """

    # initializing a vanilla transformer decoder from the (default) configuration
    decoder = VanillaTransformerDecoder(VanillaTransformerConfig())
    # moving the decoder to the default DEVICE
    decoder.to(DEVICE)

    # creating a dummy input of size (sequence_length) and (1, sequence_length, embedding_dim)
    sequence_length = 10
    embedding_dimension = decoder.config.embedding_dim
    input_ids = torch.ones((sequence_length,), device=DEVICE).long() * 2
    encoder_output = torch.ones(
        (1, sequence_length, embedding_dimension), device=DEVICE
    ).float()
    decoder_padding_mask = input_ids.eq(decoder.config.pad_token_id)

    # forwarding through the decoder
    output = decoder(
        input_ids=input_ids,
        encoder_output=encoder_output,
        padding_mask=decoder_padding_mask,
    )

    # ASSERTS

    assert isinstance(output, VanillaDecoderOutput)


def test_decoder_forward_pass_with_batched_input():
    """Tests the forward pass through a vanilla transformer decoder.
    The input is batched.
    """

    # initializing a vanilla transformer decoder from the (default) configuration
    decoder = VanillaTransformerDecoder(VanillaTransformerConfig())
    # moving the decoder to the default DEVICE
    decoder.to(DEVICE)

    # creating a dummy a input and encoder output of size (batch_size, sequence_length) and (batch_size, sequence_length, embedding_dim)
    sequence_length = 10
    batch_size = 2
    embedding_dimension = decoder.config.embedding_dim
    input_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long()
    encoder_output = torch.ones(
        (batch_size, sequence_length, embedding_dimension), device=DEVICE
    ).float()

    # forwarding through the decoder
    output = decoder(input_ids=input_ids, encoder_output=encoder_output)

    # ASSERTS

    assert isinstance(output, VanillaDecoderOutput)


def test_decoder_forward_pass_with_batched_input_and_padding_mask():
    """Tests the forward pass through a vanilla transformer decoder.
    The input is batched and the decoder padding mask is given.
    """

    # initializing a vanilla transformer decoder from the (default) configuration
    decoder = VanillaTransformerDecoder(VanillaTransformerConfig())
    # moving the decoder to the default DEVICE
    decoder.to(DEVICE)

    # creating a dummy input and encoder output of size (batch_size, sequence_length) and (batch_size, sequence_length, embedding_dim)
    sequence_length = 10
    batch_size = 2
    embedding_dimension = decoder.config.embedding_dim
    input_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long() * 2
    encoder_output = torch.ones(
        (batch_size, sequence_length, embedding_dimension), device=DEVICE
    ).float()
    decoder_padding_mask = input_ids.eq(decoder.config.pad_token_id)

    # forwarding through the decoder
    output = decoder(
        input_ids=input_ids,
        encoder_output=encoder_output,
        padding_mask=decoder_padding_mask,
    )

    # ASSERTS

    assert isinstance(output, VanillaDecoderOutput)
