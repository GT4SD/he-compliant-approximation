"""Testing the vanilla transformer."""

import os
from tempfile import TemporaryDirectory

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

# default device to run tests
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_huggingface_installation():
    """Tests the installation of the hugging face transformers package."""

    # ASSERTS

    assert transformers.__version__


def test_model_init():
    """Tests the initialization of a vanilla transformer model."""

    # initializing the (default) vanilla transformer configuration
    configuration = VanillaTransformerConfig()

    # initializing a vanilla transformer model from the configuration
    model = VanillaTransformer(configuration)

    # ASSERTS

    assert isinstance(model, VanillaTransformer)


def test_model_save_pretrained():
    """Tests the saving of a vanilla transformer model."""

    with TemporaryDirectory() as temp_dir:
        # initializing a vanilla transformer model from the (default) configuration
        model = VanillaTransformer(VanillaTransformerConfig())

        # saving the randomly initialized model and the (default) configuration
        model.save_pretrained(save_directory=temp_dir)

        # ASSERTS

        assert isinstance(model, VanillaTransformer)
        assert os.path.isdir(temp_dir)
        assert sorted(os.listdir(temp_dir)) == sorted(
            ["config.json", "pytorch_model.bin"]
        )


def test_model_load_pretrained():
    """Tests the loading of a vanilla transformer model and configuration."""

    with TemporaryDirectory() as temp_dir:
        # initializing a vanilla transformer model from the (default) configuration
        model = VanillaTransformer(VanillaTransformerConfig())

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


def test_model_forward_pass_with_not_batched_input():
    """Tests the forward pass through a vanilla transformer model.
    The input is not batched.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input and output of size (sequence_length)
    sequence_length = 10
    input_ids = torch.ones((sequence_length,), device=DEVICE).long()
    output_ids = torch.ones((sequence_length,), device=DEVICE).long()

    # forwarding through the model
    output = model(encoder_input_ids=input_ids, decoder_input_ids=output_ids)

    # ASSERTS

    assert isinstance(output, VanillaTransformerOutput)


def test_model_forward_pass_with_not_batched_input_and_padding_masks():
    """Tests the forward pass through a vanilla transformer model.
    The input is not batched and the decoder and encoder padding mask are given.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input and output of size (sequence_length)
    sequence_length = 10
    input_ids = torch.ones((sequence_length,), device=DEVICE).long() * 2
    output_ids = torch.ones((sequence_length,), device=DEVICE).long() * 2
    encoder_padding_mask = input_ids.eq(model.config.pad_token_id)
    decoder_padding_mask = output_ids.eq(model.config.pad_token_id)

    # forwarding through the model
    output = model(
        encoder_input_ids=input_ids,
        decoder_input_ids=output_ids,
        encoder_padding_mask=encoder_padding_mask,
        decoder_padding_mask=decoder_padding_mask,
    )

    # ASSERTS

    assert isinstance(output, VanillaTransformerOutput)


def test_model_forward_pass_with_batched_input():
    """Tests the forward pass through a vanilla transformer model.
    The input is batched.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input and output of size (batch_size, sequence_length)
    sequence_length = 10
    batch_size = 2
    input_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long()
    output_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long()

    # forwarding through the model
    output = model(encoder_input_ids=input_ids, decoder_input_ids=output_ids)

    # ASSERTS

    assert isinstance(output, VanillaTransformerOutput)


def test_model_forward_pass_with_batched_input_and_padding_masks():
    """Tests the forward pass through a vanilla transformer model.
    The input is batched and the decoder and encoder padding mask are given.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input and output of size (batch_size, sequence_length)
    sequence_length = 10
    batch_size = 2
    input_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long() * 2
    output_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long() * 2
    encoder_padding_mask = input_ids.eq(model.config.pad_token_id)
    decoder_padding_mask = output_ids.eq(model.config.pad_token_id)

    # forwarding through the model
    output = model(
        encoder_input_ids=input_ids,
        decoder_input_ids=output_ids,
        encoder_padding_mask=encoder_padding_mask,
        decoder_padding_mask=decoder_padding_mask,
    )

    # ASSERTS

    assert isinstance(output, VanillaTransformerOutput)


def test_model_encoder_forward_pass_with_not_batched_input():
    """Tests the ability to make forward pass through the encoder of the vanilla transformer model.
    The input is not batched.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input of size (sequence_length)
    sequence_length = 10
    input_ids = torch.ones((sequence_length,), device=DEVICE).long()

    # forwarding through the encoder
    output = model.encode(input_ids=input_ids)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)


def test_model_encoder_forward_pass_with_not_batched_input_and_padding_mask():
    """Tests the ability to make forward pass through the encoder of the vanilla transformer model.
    The input is not batched and the encoder padding mask is given.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input of size (sequence_length)
    sequence_length = 10
    input_ids = torch.ones((sequence_length,), device=DEVICE).long() * 2
    encoder_padding_mask = input_ids.eq(model.config.pad_token_id)

    # forwarding through the encoder
    output = model.encode(input_ids=input_ids, padding_mask=encoder_padding_mask)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)


def test_encoder_forward_pass_with_batched_input():
    """Tests the ability to make forward pass through the encoder of the vanilla transformer model.
    The input is batched.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input of size (batch_size, sequence_length)
    sequence_length = 10
    batch_size = 2
    input_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long()

    # forwarding through the encoder
    output = model.encode(input_ids=input_ids)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)


def test_model_encoder_forward_pass_with_batched_input_and_padding_mask():
    """Tests the ability to make forward pass through the encoder of the vanilla transformer model.
    The input is batched and the encoder padding mask is given.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input of size (batch_size, sequence_length)
    sequence_length = 10
    batch_size = 2
    input_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long() * 2
    encoder_padding_mask = input_ids.eq(model.config.pad_token_id)

    # forwarding through the encoder
    output = model.encode(input_ids=input_ids, padding_mask=encoder_padding_mask)

    # ASSERTS

    assert isinstance(output, VanillaEncoderOutput)


def test_decoder_forward_pass_with_not_batched_input():
    """Tests the ability to make forward pass through a vanilla decoder.
    The input is not batched.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input and encoder output of size (sequence_length) and (1, sequence_length, embedding_dim)
    sequence_length = 10
    embedding_dimension = model.config.embedding_dim
    input_ids = torch.ones((sequence_length,), device=DEVICE).long()
    encoder_output = torch.ones((1, sequence_length, embedding_dimension), device=DEVICE).float()

    # forwarding through the decoder
    output = model.decode(input_ids=input_ids, encoder_output=encoder_output)

    # ASSERTS

    assert isinstance(output, VanillaDecoderOutput)


def test_decoder_forward_pass_with_not_batched_input_and_padding_mask():
    """Tests the ability to make forward pass through a vanilla decoder.
    The input is not batched and the decoder padding mask is given.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input and encoder output of size (sequence_length) and (1, sequence_length, embedding_dim)
    sequence_length = 10
    embedding_dimension = model.config.embedding_dim
    input_ids = torch.ones((sequence_length,), device=DEVICE).long() * 2
    encoder_output = torch.ones((1, sequence_length, embedding_dimension), device=DEVICE).float()
    decoder_padding_mask = input_ids.eq(model.config.pad_token_id)

    # forwarding through the decoder
    output = model.decode(
        input_ids=input_ids,
        encoder_output=encoder_output,
        padding_mask=decoder_padding_mask,
    )

    # ASSERTS

    assert isinstance(output, VanillaDecoderOutput)


def test_decoder_forward_pass_with_batched_input():
    """Tests the ability to make forward pass through a vanilla decoder of the vanilla transformer model.
    The input is batched.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input and encoder output of size (batch_size, sequence_length) and (batch_size, sequence_length, embedding_dim)
    sequence_length = 10
    batch_size = 2
    embedding_dimension = model.config.embedding_dim
    input_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long()
    encoder_output = torch.ones(
        (batch_size, sequence_length, embedding_dimension), device=DEVICE
    ).float()

    # forwarding through the decoder
    output = model.decode(input_ids=input_ids, encoder_output=encoder_output)

    # ASSERTS

    assert isinstance(output, VanillaDecoderOutput)


def test_decoder_forward_pass_with_batched_input_and_padding_mask():
    """Tests the ability to make forward pass through a vanilla decoder of the vanilla transformer model.
    The input is batched and the decoder padding mask is given.
    """

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input and encoder output of size (batch_size, sequence_length) and (batch_size, sequence_length, embedding_dim)
    sequence_length = 10
    batch_size = 2
    embedding_dimension = model.config.embedding_dim
    input_ids = torch.ones((batch_size, sequence_length), device=DEVICE).long() * 2
    encoder_output = torch.ones(
        (batch_size, sequence_length, embedding_dimension), device=DEVICE
    ).float()
    decoder_padding_mask = input_ids.eq(model.config.pad_token_id)

    # forwarding through the decoder
    output = model.decode(
        input_ids=input_ids,
        encoder_output=encoder_output,
        padding_mask=decoder_padding_mask,
    )

    # ASSERTS

    assert isinstance(output, VanillaDecoderOutput)


def test_get_encoder():
    """Tests the ability to return the encoder module of the vanilla transformer model."""

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())

    # getting the encoder module
    encoder = model.get_encoder()

    # ASSERTS

    assert isinstance(encoder, VanillaTransformerEncoder)


def test_get_decoder():
    """Tests the ability to return the decoder module of the vanilla transformer model."""

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())

    # getting the decoder module
    encoder = model.get_decoder()

    # ASSERTS

    assert isinstance(encoder, VanillaTransformerDecoder)


def test_generate_greedy_search():
    """Tests the ability to generate of a vanilla transformer model, using greedy search."""

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input of size (1, sequence_length)
    sequence_length = 10
    input_ids = torch.ones((1, sequence_length), device=DEVICE).long()

    # generating the sequence
    max_length = 30
    output = model.generate(input_ids, do_sample=False, max_length=max_length)

    # ASSERTS

    assert isinstance(output, torch.Tensor)
    assert output.size()[1] > 1
    assert output.size()[1] <= max_length


def test_generate_beam_search():
    """Tests the ability to generate of a vanilla transformer model, using beam search."""

    # initializing a vanilla transformer model from the (default) configuration
    model = VanillaTransformer(VanillaTransformerConfig())
    # moving the model to the default DEVICE
    model.to(DEVICE)

    # creating a dummy input of size (1, sequence_length)
    sequence_length = 10
    input_ids = torch.ones((1, sequence_length), device=DEVICE).long()

    # generating the sequence
    max_length = 30
    num_beams = 5
    output = model.generate(
        input_ids, do_sample=False, max_length=max_length, num_beams=num_beams
    )

    # ASSERTS

    assert isinstance(output, torch.Tensor)
    assert output.size()[1] > 1
    assert output.size()[1] <= max_length
