"""Testing the LeNet model."""

import pytest
import torch
from torch import Tensor

from hela.models.lenet.configuration import LeNetConfig
from hela.models.lenet.model import LeNet


@pytest.mark.parametrize(
    "lenet_type,num_classes,greyscale",
    [
        (lenet_type, num_classes, greyscale)
        for lenet_type in ["lenet-1", "lenet-4", "lenet-5"]
        for num_classes in [10, 100, 1000]
        for greyscale in [True, False]
    ],
    ids=[
        f"{lenet_type} - {num_classes} classes - {'greyscale' if greyscale else 'color'}"
        for lenet_type in ["lenet-1", "lenet-4", "lenet-5"]
        for num_classes in [10, 100, 1000]
        for greyscale in [True, False]
    ],
)
def test_lenet_config(lenet_type, num_classes, greyscale):
    """Tests the LeNet configuration with parametrized arguments."""

    # Creating a LeNet configuration with the parametrized arguments
    config = LeNetConfig(
        lenet_type=lenet_type, num_classes=num_classes, greyscale=greyscale
    )

    # ASSERTS

    assert (
        config.lenet_type == lenet_type
    ), "The LeNet type does not match the expected value."
    assert (
        config.num_classes == num_classes
    ), "The number of classes does not match the expected value."
    expected_in_channels = 1 if greyscale else 3
    assert (
        config.in_channels == expected_in_channels
    ), f"The in_channels value {config.in_channels} does not match the expected value {expected_in_channels} based on the greyscale flag."


@pytest.mark.parametrize(
    "lenet_type,num_classes,greyscale",
    [
        (lenet_type, num_classes, greyscale)
        for lenet_type in ["lenet-1", "lenet-4", "lenet-5"]
        for num_classes in [10, 100, 1000]
        for greyscale in [True, False]
    ],
    ids=[
        f"{lenet_type} - {num_classes} classes - {'greyscale' if greyscale else 'color'}"
        for lenet_type in ["lenet-1", "lenet-4", "lenet-5"]
        for num_classes in [10, 100, 1000]
        for greyscale in [True, False]
    ],
)
def test_lenet_model_init(lenet_type, num_classes, greyscale):
    """Tests the initialization of LeNet model with parametrized arguments."""

    # creating a LeNet configuration with the parametrized arguments
    config = LeNetConfig(
        lenet_type=lenet_type, num_classes=num_classes, greyscale=greyscale
    )

    # initializing LeNet model from the configuration
    model = LeNet(config)

    # ASSERTS

    assert isinstance(model, LeNet), "The model is not an instance of LeNet."
    assert (
        model.config == config
    ), "The model configuration does not match the expected configuration."


@pytest.mark.parametrize(
    "lenet_type,num_classes,greyscale,batch_size",
    [
        (lenet_type, num_classes, greyscale, batch_size)
        for lenet_type in ["lenet-1", "lenet-4", "lenet-5"]
        for num_classes in [10, 100, 1000]
        for greyscale in [True, False]
        for batch_size in [1, 10]
    ],
    ids=[
        f"{lenet_type} - {num_classes} classes - {'greyscale' if greyscale else 'color'}{' - batched' if batch_size > 1 else ''}"
        for lenet_type in ["lenet-1", "lenet-4", "lenet-5"]
        for num_classes in [10, 100, 1000]
        for greyscale in [True, False]
        for batch_size in [1, 10]
    ],
)
def test_model_forward_pass_with_varying_batch_sizes(
    lenet_type, num_classes, greyscale, batch_size
):
    """Tests the forward pass through LeNet model with varying batch sizes."""

    # determining the image size based on lenet_type
    image_size = 28 if lenet_type == "lenet-1" else 32

    # initializing LeNet model from the parametrized configuration
    config = LeNetConfig(
        lenet_type=lenet_type, num_classes=num_classes, greyscale=greyscale
    )
    model = LeNet(config)

    # creating a dummy input of size (batch_size, C, H, W)
    channels = 1 if greyscale else 3
    input_tensor = torch.ones(
        (batch_size, channels, image_size, image_size), dtype=torch.float
    )

    # forwarding through the model
    output = model(input_tensor)  # Input tensor already includes batch dimension

    # ASSERTS

    assert isinstance(output, Tensor), "Output is not a tensor."
    assert (
        output.size(0) == batch_size
    ), f"Output batch size {output.size(0)} does not match input batch size {batch_size}."
