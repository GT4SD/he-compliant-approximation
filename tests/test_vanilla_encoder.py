"""Testing the vanilla transformer encoder."""

import os
from tempfile import TemporaryDirectory

import pytest
import torch

from hela.models.vanilla_transformer.configuration import VanillaTransformerConfig
from hela.models.vanilla_transformer.model import (
    VanillaEncoderOutput,
    VanillaTransformerEncoder,
)

# defining the devices to run the tests on
DEVICE_LIST = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f"device: {device}" for device in DEVICE_LIST],
)
def test_encoder_init(device: str):
    """Tests the initialization of a vanilla transformer encoder."""

    # initializing the (default) configuration
    configuration = VanillaTransformerConfig(device=device)

    # initializing a vanilla transformer encoder model from the configuration
    model = VanillaTransformerEncoder(configuration)

    # ASSERTS

    assert isinstance(model, VanillaTransformerEncoder)


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f"device: {device}" for device in DEVICE_LIST],
)
def test_encoder_save_pretrained(device: str):
    """Tests the saving of a vanilla transformer encoder model and configuration."""

    with TemporaryDirectory() as temp_dir:
        # initializing a vanilla transformer encoder from the (default) configuration
        encoder = VanillaTransformerEncoder(VanillaTransformerConfig(device=device))

        # saving the randomly initialized model and the (default) configuration
        encoder.save_pretrained(save_directory=temp_dir)

        # ASSERTS

        assert isinstance(encoder, VanillaTransformerEncoder)
        assert os.path.isdir(temp_dir)
        assert sorted(os.listdir(temp_dir)) == sorted(
            ["config.json", "model.safetensors"]
        )


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f"device: {device}" for device in DEVICE_LIST],
)
def test_encoder_load_pretrained(device: str):
    """Tests the loading of a vanilla transformer encoder model and configuration."""

    with TemporaryDirectory() as temp_dir:
        # initializing a vanilla transformer encoder from the (default) configuration
        encoder = VanillaTransformerEncoder(VanillaTransformerConfig(device=device))

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


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f"device: {device}" for device in DEVICE_LIST],
)
def test_encoder_forward_pass_with_no_batched_input(device: str):
    """Tests the forward pass through a vanilla transformer encoder.
    The input is not batched.
    """

    # initializing a vanilla transformer encoder from the (default) configuration
    encoder = VanillaTransformerEncoder(VanillaTransformerConfig(device=device))
    # moving the encoder to the default device
    encoder.to(device)

    # creating a dummy input of size (sequence_length)
    sequence_length = 10
    input_ids = torch.ones((sequence_length,), device=device).long()

    # forwarding through the encoder
    output = encoder(input_ids=input_ids)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f"device: {device}" for device in DEVICE_LIST],
)
def test_encoder_forward_pass_with_no_batched_input_and_padding_mask(device: str):
    """Tests the forward pass through a vanilla transformer encoder.
    The input is batched and the encoder padding mask is given.
    """

    # initializing a vanilla transformer encoder from the (default) configuration
    encoder = VanillaTransformerEncoder(VanillaTransformerConfig(device=device))
    # moving the encoder to the default device
    encoder.to(device)

    # creating a dummy input of size (sequence_length)
    sequence_length = 10
    input_ids = torch.ones((sequence_length,), device=device).long() * 2
    encoder_padding_mask = input_ids.eq(encoder.config.pad_token_id)

    # forwarding through the encoder
    output = encoder(input_ids=input_ids, padding_mask=encoder_padding_mask)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f"device: {device}" for device in DEVICE_LIST],
)
def test_encoder_forward_pass_with_batched_input(device: str):
    """Tests the forward pass through a vanilla transformer encoder.
    The input is batched.
    """

    # initializing a vanilla transformer encoder from the (default) configuration
    encoder = VanillaTransformerEncoder(VanillaTransformerConfig(device=device))
    # moving the encoder to the default device
    encoder.to(device)

    # creating a dummy input of size (batch_size, sequence_length)
    sequence_length = 10
    batch_size = 2
    input_ids = torch.ones((batch_size, sequence_length), device=device).long()

    # forwarding through the encoder
    output = encoder(input_ids=input_ids)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f"device: {device}" for device in DEVICE_LIST],
)
def test_encoder_forward_pass_with_batched_input_and_padding_mask(device: str):
    """Tests the forward pass through a vanilla transformer encoder.
    The input is batched and the encoder padding mask is given.
    """

    # initializing a vanilla transformer encoder from the (default) configuration
    encoder = VanillaTransformerEncoder(VanillaTransformerConfig(device=device))
    # moving the encoder to the default device
    encoder.to(device)

    # creating a dummy input of size (batch_size, sequence_length)
    sequence_length = 10
    batch_size = 2
    input_ids = torch.ones((batch_size, sequence_length), device=device).long() * 2
    encoder_padding_mask = input_ids.eq(encoder.config.pad_token_id)

    # forwarding through the encoder
    output = encoder(input_ids=input_ids, padding_mask=encoder_padding_mask)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()
