"""Testing the SqueezeNet model."""

import pytest
import torch
from torch import Tensor

from hela.models.squeezenet.configuration import SqueezeNetConfig
from hela.models.squeezenet.model import SqueezeNet

# defining the devices to run the tests on
DEVICE_LIST = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
# defining the values for the number of classes parameter
NUM_CLASSES_LIST = [10, 100]
# defining the values for the dropout rate parameter
DROPOUT_LIST = [0.2, 0.5]
# defining the values for the batch size parameter
BATCH_SIZE_LIST = [1, 2]

# defining channels, height and width of the input used in some tests
TEST_INPUT_CHANNELS = 3
TEST_INPUT_HEIGHT = TEST_INPUT_WIDTH = 224


@pytest.mark.parametrize(
    "num_classes,dropout",
    [
        (num_classes, dropout)
        for num_classes in NUM_CLASSES_LIST
        for dropout in DROPOUT_LIST
    ],
    ids=[
        f" num_classes: {num_classes} - dropout: {dropout} "
        for num_classes in NUM_CLASSES_LIST
        for dropout in DROPOUT_LIST
    ],
)
def test_squeezenet_config(num_classes: int, dropout: float):
    """
    Tests the SqueezeNet configuration with parametrized arguments.

    Args:
        num_classes: number of classes for the model.
        dropout: dropout rate for the model.
    """

    # creating a SqueezeNet configuration with the parametrized arguments
    config = SqueezeNetConfig(num_classes=num_classes, dropout=dropout)

    # ASSERTS

    assert (
        config.num_classes == num_classes
    ), "The number of classes does not match the expected value."
    assert (
        config.dropout == dropout
    ), "The dropout rate does not match the expected value."


@pytest.mark.parametrize(
    "num_classes,dropout",
    [
        (num_classes, dropout)
        for num_classes in NUM_CLASSES_LIST
        for dropout in DROPOUT_LIST
    ],
    ids=[
        f" num_classes: {num_classes} - dropout: {dropout} "
        for num_classes in NUM_CLASSES_LIST
        for dropout in DROPOUT_LIST
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
    "num_classes,dropout,batch_size,device",
    [
        (num_classes, dropout, batch_size, device)
        for num_classes in NUM_CLASSES_LIST
        for dropout in DROPOUT_LIST
        for batch_size in BATCH_SIZE_LIST
        for device in DEVICE_LIST
    ],
    ids=[
        f" num_classes: {num_classes} - dropout: {dropout} - batch_size: {batch_size} - device: {device} "
        for num_classes in NUM_CLASSES_LIST
        for dropout in DROPOUT_LIST
        for batch_size in BATCH_SIZE_LIST
        for device in DEVICE_LIST
    ],
)
def test_model_forward_pass(
    num_classes: int, dropout: float, batch_size: int, device: str
):
    """Tests the forward pass through SqueezeNet model with parametrized arguments.

    Args:
        num_classes: number of classes for the model.
        dropout: dropout rate for the model.
        batch_size: size of the batch for the input.
        device: device to run the model on.
    """

    # creating a SqueezeNet configuration with the parametrized arguments
    config = SqueezeNetConfig(num_classes=num_classes, dropout=dropout)

    # initializing SqueezeNet model from the configuration
    model = SqueezeNet(config).to(device)

    # creating a dummy input of size (batch_size, TEST_INPUT_CHANNELS, TEST_INPUT_HEIGHT, TEST_INPUT_WIDTH)
    input_tensor = torch.ones(
        (batch_size, TEST_INPUT_CHANNELS, TEST_INPUT_HEIGHT, TEST_INPUT_WIDTH),
        dtype=torch.float,
        device=device,
    )

    # forwarding through the model
    output = model(input_tensor)

    # ASSERTS

    assert isinstance(output, Tensor), "Output is not a tensor."
    assert (
        output.size(0) == batch_size
    ), f"Output batch size {output.size(0)} does not match input batch size {batch_size}."

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()
