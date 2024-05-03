"""Testing the vanilla transformer."""

import os
from tempfile import TemporaryDirectory

import pytest
import torch
import transformers

from hela.models.vanilla_transformer.configuration import VanillaTransformerConfig
from hela.models.vanilla_transformer.model import (
    VanillaDecoderOutput,
    VanillaEncoderOutput,
    VanillaTransformer,
    VanillaTransformerDecoder,
    VanillaTransformerEncoder,
    VanillaTransformerOutput,
)

# defining the sequence length of the input used in some tests
SEQUENCE_LENGTH = 10
# defining the devices to run the tests on
DEVICE_LIST = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
# defining the values for the batch size parameter
BATCH_SIZE_LIST = [1, 2]


def test_huggingface_installation():
    """Tests the installation of the hugging face transformers package."""

    # ASSERTS

    assert transformers.__version__


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f" device: {device} " for device in DEVICE_LIST],
)
def test_model_init(device: str):
    """
    Tests the initialization of a vanilla transformer model.

    Args:
        device: device on which the model will be initialized.
    """

    # initializing the (default) vanilla transformer configuration
    configuration = VanillaTransformerConfig(device=device)

    # initializing a vanilla transformer model from the configuration
    model = VanillaTransformer(configuration)

    # ASSERTS

    assert isinstance(model, VanillaTransformer)

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f" device: {device} " for device in DEVICE_LIST],
)
def test_model_save_pretrained(device: str):
    """
    Tests the saving of a vanilla transformer model.

    Args:
        device: device on which the model will be initialized.
    """

    with TemporaryDirectory() as temp_dir:
        # initializing a vanilla transformer model from the (default) configuration
        model = VanillaTransformer(VanillaTransformerConfig(device=device))

        # saving the randomly initialized model and the (default) configuration
        model.save_pretrained(save_directory=temp_dir)

        # ASSERTS

        assert isinstance(model, VanillaTransformer)
        assert os.path.isdir(temp_dir)
        assert sorted(os.listdir(temp_dir)) == sorted(
            ["config.json", "generation_config.json", "model.safetensors"]
        )

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f" device: {device} " for device in DEVICE_LIST],
)
def test_model_load_pretrained(device: str):
    """
    Tests the loading of a vanilla transformer model and configuration.

    Args:
        device: device on which the model will be initialized.
    """

    with TemporaryDirectory() as temp_dir:
        # initializing a vanilla transformer model from the (default) configuration
        model = VanillaTransformer(VanillaTransformerConfig(device=device))

        # saving the randomly initialized model and the (default) configuration
        model.save_pretrained(save_directory=temp_dir)

        # loading the saved model and configuration
        loaded_model = VanillaTransformer.from_pretrained(temp_dir)

        # ASSERTS

        assert isinstance(model, VanillaTransformer)
        assert isinstance(loaded_model, VanillaTransformer)

        for name, param in model.named_parameters():
            value = loaded_model
            for attr in name.split("."):
                value = getattr(value, attr)
            assert torch.equal(param, value)

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "batch_size,use_padding_mask,device",
    [
        (batch_size, use_padding_mask, device)
        for batch_size in BATCH_SIZE_LIST
        for use_padding_mask in [True, False]
        for device in DEVICE_LIST
    ],
    ids=[
        f"batch_size: {batch_size}, use_padding_mask: {use_padding_mask}, device: {device}"
        for batch_size in BATCH_SIZE_LIST
        for use_padding_mask in [True, False]
        for device in DEVICE_LIST
    ],
)
def test_model_forward_pass(batch_size: int, use_padding_mask: bool, device: str):
    """
    Tests the forward pass through a vanilla transformer model.

    Args:
        batch_size: size of the batch for the input.
        use_padding_mask: whether to use padding mask or not.
        device: device to run the model on.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig(device=device))
    # moving the model to the default device
    model.to(device)

    # creating a dummy input and output of size (batch_size, SEQUENCE_LENGTH)
    input_ids = torch.ones((batch_size, SEQUENCE_LENGTH), device=device).long() * 2
    output_ids = torch.ones((batch_size, SEQUENCE_LENGTH), device=device).long() * 2

    if use_padding_mask:
        encoder_padding_mask = input_ids.eq(model.config.pad_token_id)
        decoder_padding_mask = output_ids.eq(model.config.pad_token_id)
    else:
        encoder_padding_mask = None
        decoder_padding_mask = None

    # forwarding through the model
    output = model(
        encoder_input_ids=input_ids,
        decoder_input_ids=output_ids,
        encoder_padding_mask=encoder_padding_mask,
        decoder_padding_mask=decoder_padding_mask,
    )

    # ASSERTS

    assert isinstance(output, VanillaTransformerOutput)

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "batch_size,use_padding_mask,device",
    [
        (batch_size, use_padding_mask, device)
        for batch_size in BATCH_SIZE_LIST
        for use_padding_mask in [True, False]
        for device in DEVICE_LIST
    ],
    ids=[
        f"batch_size: {batch_size}, use_padding_mask: {use_padding_mask}, device: {device}"
        for batch_size in BATCH_SIZE_LIST
        for use_padding_mask in [True, False]
        for device in DEVICE_LIST
    ],
)
def test_encoder_forward_pass(batch_size: int, use_padding_mask: bool, device: str):
    """
    Tests the ability to make forward pass through the encoder of the vanilla transformer model.

    Args:
        batch_size: size of the batch for the input.
        use_padding_mask: whether to use padding mask or not.
        device: device to run the model on.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig(device=device))
    # moving the model to the default device
    model.to(device)

    # creating a dummy input of size (batch_size, SEQUENCE_LENGTH)

    input_ids = torch.ones((batch_size, SEQUENCE_LENGTH), device=device).long() * 2

    if use_padding_mask:
        encoder_padding_mask = input_ids.eq(model.config.pad_token_id)
    else:
        encoder_padding_mask = None

    # forwarding through the encoder
    output = model.encode(input_ids=input_ids, padding_mask=encoder_padding_mask)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)
    assert output.last_hidden_state.shape == (
        batch_size,
        SEQUENCE_LENGTH,
        model.config.embedding_dim,
    )

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "batch_size,use_padding_mask,device",
    [
        (batch_size, use_padding_mask, device)
        for batch_size in BATCH_SIZE_LIST
        for use_padding_mask in [True, False]
        for device in DEVICE_LIST
    ],
    ids=[
        f" batch_size: {batch_size} - use_padding_mask: {use_padding_mask} - device: {device} "
        for batch_size in BATCH_SIZE_LIST
        for use_padding_mask in [True, False]
        for device in DEVICE_LIST
    ],
)
def test_decoder_forward_pass(batch_size: int, use_padding_mask: bool, device: str):
    """Tests the ability to make forward pass through a vanilla decoder of the vanilla transformer model.

    Args:
        batch_size: size of the batch for the input.
        use_padding_mask: whether to use padding mask or not.
        device: the device on which to run the tests.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig(device=device))
    # moving the model to the device
    model.to(device)

    # creating a dummy input and encoder output of size (batch_size, SEQUENCE_LENGTH) and (batch_size, SEQUENCE_LENGTH, embedding_dim)

    embedding_dimension = model.config.embedding_dim

    input_ids = torch.ones((batch_size, SEQUENCE_LENGTH), device=device).long() * 2
    encoder_output = torch.ones(
        (batch_size, SEQUENCE_LENGTH, embedding_dimension), device=device
    ).float()

    if use_padding_mask:
        decoder_padding_mask = input_ids.eq(model.config.pad_token_id)
    else:
        decoder_padding_mask = None

    # forwarding through the decoder
    output = model.decode(
        input_ids=input_ids,
        encoder_output=encoder_output,
        padding_mask=decoder_padding_mask,
    )

    # ASSERTS

    assert isinstance(output, VanillaDecoderOutput)
    assert output.last_hidden_state.shape == (
        batch_size,
        SEQUENCE_LENGTH,
        model.config.embedding_dim,
    )

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f" device: {device} " for device in DEVICE_LIST],
)
def test_get_encoder(device: str):
    """
    Tests the ability to return the encoder module of the vanilla transformer model.

    Args:
        device: device on which the model will be initialized.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig(device=device))

    # getting the encoder module
    encoder = model.get_encoder()

    # ASSERTS

    assert isinstance(encoder, VanillaTransformerEncoder)

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f" device: {device} " for device in DEVICE_LIST],
)
def test_get_decoder(device: str):
    """
    Tests the ability to return the decoder module of the vanilla transformer model.

    Args:
        device: device on which the model will be initialized.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig(device=device))

    # getting the decoder module
    encoder = model.get_decoder()

    # ASSERTS

    assert isinstance(encoder, VanillaTransformerDecoder)

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "device, beam_search",
    [(device, beam_search) for device in DEVICE_LIST for beam_search in [False, True]],
    ids=[
        f" device: {device} - beam_search: {beam_search} "
        for device in DEVICE_LIST
        for beam_search in [False, True]
    ],
)
def test_generate_greedy_search(device: str, beam_search: bool):
    """
    Tests the ability to generate of a vanilla transformer model, using greedy search.

    Args:
        device: device on which the model will be initialized.
        beam_search: whether to use beam search or not.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig(device=device))
    # moving the model to the default device
    model.to(device)

    # creating a dummy input of size (1, SEQUENCE_LENGTH)

    input_ids = torch.ones((1, SEQUENCE_LENGTH), device=device).long()
    num_beams = 3 if beam_search else 1

    # generating the sequence
    max_length = 30
    output = model.generate(
        input_ids,
        do_sample=False,
        max_length=max_length,
        num_beams=num_beams,
    )

    # ASSERTS

    assert isinstance(output, torch.Tensor)
    assert output.size()[1] > 1
    assert output.size()[1] <= max_length

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()
