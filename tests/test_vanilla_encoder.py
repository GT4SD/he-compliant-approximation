"""Testing the vanilla transformer encoder."""

import os
from tempfile import TemporaryDirectory

import torch

from hela.models.vanilla_transformer.configuration import VanillaTransformerConfig
from hela.models.vanilla_transformer.model import (
    VanillaEncoderOutput,
    VanillaTransformerEncoder,
)

# default device to run the tests
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_encoder_init():
    """Tests the initialization of a vanilla transformer encoder."""

    # initializing the (default) configuration
    configuration = VanillaTransformerConfig()

    # initializing a vanilla transformer encoder model from the configuration
    model = VanillaTransformerEncoder(configuration)

    # ASSERTS

    assert isinstance(model, VanillaTransformerEncoder)


def test_encoder_save_pretrained():
    """Tests the saving of a vanilla transformer encoder model and configuration."""

    with TemporaryDirectory() as temp_dir:
        # initializing a vanilla transformer encoder from the (default) configuration
        encoder = VanillaTransformerEncoder(VanillaTransformerConfig())

        # saving the randomly initialized model and the (default) configuration
        encoder.save_pretrained(save_directory=temp_dir)

        # ASSERTS

        assert isinstance(encoder, VanillaTransformerEncoder)
        assert os.path.isdir(temp_dir)
        assert sorted(os.listdir(temp_dir)) == sorted(
            ["config.json", "pytorch_model.bin"]
        )


def test_encoder_load_pretrained():
    """Tests the loading of a vanilla transformer encoder model and configuration."""

    with TemporaryDirectory() as temp_dir:
        # initializing a vanilla transformer encoder from the (default) configuration
        encoder = VanillaTransformerEncoder(VanillaTransformerConfig())

        # saving the randomly initialized model and the (default) configuration
        encoder.save_pretrained(save_directory=temp_dir)

        # loading the saved model and configuration
        loaded_encoder = VanillaTransformerEncoder.from_pretrained(temp_dir)

        # ASSERTS

        assert isinstance(encoder, VanillaTransformerEncoder)
        assert isinstance(loaded_encoder, VanillaTransformerEncoder)

        for name, param in encoder.named_parameters():
            value = loaded_encoder
            for attr in name.split("."):
                value = getattr(value, attr)
            assert torch.equal(param, value)


def test_encoder_forward_pass_with_no_batched_input():
    """Tests the forward pass through a vanilla transformer encoder.
    The input is not batched.
    """

    # initializing a vanilla transformer encoder from the (default) configuration
    encoder = VanillaTransformerEncoder(VanillaTransformerConfig())
    # moving the encoder to the default DEVICE
    encoder.to(DEVICE)

    # creating a dummy input of size (sequence_length)
    sequence_length = 10
    input_ids = torch.ones((sequence_length,), device=DEVICE).long()

    # forwarding through the encoder
    output = encoder(input_ids=input_ids)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)


def test_encoder_forward_pass_with_no_batched_input_and_padding_mask():
    """Tests the forward pass through a vanilla transformer encoder.
    The input is batched and the encoder padding mask is given.
    """

    # initializing a vanilla transformer encoder from the (default) configuration
    encoder = VanillaTransformerEncoder(VanillaTransformerConfig())
    # moving the encoder to the default DEVICE
    encoder.to(DEVICE)

    # creating a dummy input of size (sequence_length)
    sequence_length = 10
    input_ids = torch.ones((sequence_length,), device=DEVICE).long() * 2
    encoder_padding_mask = input_ids.eq(encoder.config.pad_token_id)

    # forwarding through the encoder
    output = encoder(input_ids=input_ids, padding_mask=encoder_padding_mask)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)


def test_encoder_forward_pass_with_batched_input():
    """Tests the forward pass through a vanilla transformer encoder.
    The input is batched.
    """

    # initializing a vanilla transformer encoder from the (default) configuration
    encoder = VanillaTransformerEncoder(VanillaTransformerConfig())
    # moving the encoder to the default DEVICE
    encoder.to(DEVICE)

    # creating a dummy input of size (batch_size, sequence_length)
    sequence_length = 10
    batch_size = 2
    input_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long()

    # forwarding through the encoder
    output = encoder(input_ids=input_ids)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)


def test_encoder_forward_pass_with_batched_input_and_padding_mask():
    """Tests the forward pass through a vanilla transformer encoder.
    The input is batched and the encoder padding mask is given.
    """

    # initializing a vanilla transformer encoder from the (default) configuration
    encoder = VanillaTransformerEncoder(VanillaTransformerConfig())
    # moving the encoder to the default DEVICE
    encoder.to(DEVICE)

    # creating a dummy input of size (batch_size, sequence_length)
    sequence_length = 10
    batch_size = 2
    input_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long() * 2
    encoder_padding_mask = input_ids.eq(encoder.config.pad_token_id)

    # forwarding through the encoder
    output = encoder(input_ids=input_ids, padding_mask=encoder_padding_mask)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)
