"""Testing the SqueezeNet model."""

import pytest
import torch
from torch import Tensor

from hela.models.squeezenet.configuration import SqueezeNetConfig
from hela.models.squeezenet.model import SqueezeNet


@pytest.mark.parametrize(
    "num_classes, dropout",
    [
        (num_classes, dropout)
        for num_classes in [10, 100, 1000]
        for dropout in [0.2, 0.5]
    ],
    ids=[
        f"{num_classes} classes - dropout={dropout}"
        for num_classes in [10, 100, 1000]
        for dropout in [0.2, 0.5]
    ],
)
def test_squeezenet_config(num_classes: int, dropout: float):
    """
    Tests the SqueezeNet configuration with parametrized arguments.

    Args:
        num_classes: number of classes for the model.
        dropout: dropout rate for the model.
    """

    # Creating a SqueezeNet configuration with the parametrized arguments
    config = SqueezeNetConfig(num_classes=num_classes, dropout=dropout)

    # ASSERTS

    assert (
        config.num_classes == num_classes
    ), "The number of classes does not match the expected value."
    assert (
        config.dropout == dropout
    ), "The dropout rate does not match the expected value."


@pytest.mark.parametrize(
    "num_classes, dropout",
    [
        (num_classes, dropout)
        for num_classes in [10, 100, 1000]
        for dropout in [0.2, 0.5]
    ],
    ids=[
        f"{num_classes} classes - dropout={dropout}"
        for num_classes in [10, 100, 1000]
        for dropout in [0.2, 0.5]
    ],
)
def test_squeezenet_model_init(num_classes: int, dropout: float):
    """
    Tests the initialization of SqueezeNet model with parametrized arguments.

    Args:
        num_classes: number of classes for the model.
        dropout: dropout rate for the model.
    """

    # creating a SqueezeNet configuration with the parametrized arguments
    config = SqueezeNetConfig(num_classes=num_classes, dropout=dropout)

    # initializing SqueezeNet model from the configuration
    model = SqueezeNet(config)

    # ASSERTS

    assert isinstance(model, SqueezeNet), "The model is not an instance of SqueezeNet."
    assert (
        model.config == config
    ), "The model configuration does not match the expected configuration."


@pytest.mark.parametrize(
    "num_classes, dropout, batch_size",
    [
        (num_classes, dropout, batch_size)
        for num_classes in [10, 100, 1000]
        for dropout in [0.2, 0.5]
        for batch_size in [1, 10]
    ],
    ids=[
        f"{num_classes} classes - dropout={dropout} - batch_size={batch_size}"
        for num_classes in [10, 100, 1000]
        for dropout in [0.2, 0.5]
        for batch_size in [1, 10]
    ],
)
def test_model_forward_pass_with_varying_batch_sizes(
    num_classes: int, dropout: float, batch_size: int
):
    """Tests the forward pass through SqueezeNet model with varying batch sizes.

    Args:
        num_classes: number of classes for the model.
        dropout: dropout rate for the model.
        batch_size: size of the batch for the input.
    """

    # initializing SqueezeNet model from the parametrized configuration
    config = SqueezeNetConfig(num_classes=num_classes, dropout=dropout)
    model = SqueezeNet(config)

    # creating a dummy input of size (batch_size, C, H, W)
    channels = 3
    image_size = 224
    input_tensor = torch.ones(
        (batch_size, channels, image_size, image_size), dtype=torch.float
    )

    # forwarding through the model
    output = model(input_tensor)

    # ASSERTS

    assert isinstance(output, Tensor), "Output is not a tensor."
    assert (
        output.size(0) == batch_size
    ), f"Output batch size {output.size(0)} does not match input batch size {batch_size}."
