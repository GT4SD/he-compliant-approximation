"""Testing the LeNet model."""

import pytest
import torch
from torch import Tensor

from henets.models.lenet.configuration import LeNetConfig
from henets.models.lenet.model import LeNet

# defining the devices to run the tests on
DEVICE_LIST = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
# defining the types of LeNet to run the tests on
LENET_TYPE_LIST = ["lenet-1", "lenet-4", "lenet-5"]
# defining the values for the number of classes parameter
NUM_CLASSES_LIST = [10, 100]
# defining the values for the batch size parameter
BATCH_SIZE_LIST = [1, 2]


@pytest.mark.parametrize(
    "lenet_type,num_classes,greyscale",
    [
        (lenet_type, num_classes, greyscale)
        for lenet_type in LENET_TYPE_LIST
        for num_classes in NUM_CLASSES_LIST
        for greyscale in [True, False]
    ],
    ids=[
        f" {lenet_type} - num_classes: {num_classes} - {'greyscale' if greyscale else 'color'} "
        for lenet_type in LENET_TYPE_LIST
        for num_classes in NUM_CLASSES_LIST
        for greyscale in [True, False]
    ],
)
def test_lenet_config(lenet_type: str, num_classes: int, greyscale: bool):
    """
    Tests the LeNet configuration with parametrized arguments.

    Args:
        lenet_type: type of LeNet model to test.
        num_classes: number of classes for the model.
        greyscale: whether the model should use greyscale input.
    """

    # creating a LeNet configuration with the parametrized arguments
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
        for lenet_type in LENET_TYPE_LIST
        for num_classes in NUM_CLASSES_LIST
        for greyscale in [True, False]
    ],
    ids=[
        f" {lenet_type} - num_classes: {num_classes} - {'greyscale' if greyscale else 'color'} "
        for lenet_type in LENET_TYPE_LIST
        for num_classes in NUM_CLASSES_LIST
        for greyscale in [True, False]
    ],
)
def test_lenet_model_init(lenet_type: str, num_classes: int, greyscale: bool):
    """
    Tests the initialization of LeNet model with parametrized arguments.

    Args:
        lenet_type: type of LeNet model to test.
        num_classes: number of classes for the model.
        greyscale: whether the model should use greyscale input.
    """

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
    "lenet_type,num_classes,greyscale,batch_size,device",
    [
        (lenet_type, num_classes, greyscale, batch_size, device)
        for lenet_type in LENET_TYPE_LIST
        for num_classes in NUM_CLASSES_LIST
        for greyscale in [True, False]
        for batch_size in BATCH_SIZE_LIST
        for device in DEVICE_LIST
    ],
    ids=[
        f" {lenet_type} - num_classes: {num_classes} - {'greyscale' if greyscale else 'color'}{' - batched' if batch_size > 1 else ' - unbatched'} - device: {device} "
        for lenet_type in LENET_TYPE_LIST
        for num_classes in NUM_CLASSES_LIST
        for greyscale in [True, False]
        for batch_size in BATCH_SIZE_LIST
        for device in DEVICE_LIST
    ],
)
def test_model_forward_pass(
    lenet_type: str, num_classes: int, greyscale: bool, batch_size: int, device: str
):
    """
    Tests the forward pass through LeNet model with parametrized arguments.

    Args:
        lenet_type: type of LeNet model to test.
        num_classes: number of classes for the model.
        greyscale: whether the model should use greyscale input.
        batch_size: size of the batch for the input.
        device: device to run the model on.
    """

    # creating a LeNet configuration with the parametrized arguments
    config = LeNetConfig(
        lenet_type=lenet_type, num_classes=num_classes, greyscale=greyscale
    )

    # initializing LeNet model from the configuration
    model = LeNet(config).to(device)

    # determining the channels based on greyscale parameter value
    channels = 1 if greyscale else 3
    # determining the image size based on lenet_type
    image_height = image_width = 28 if lenet_type == "lenet-1" else 32

    # creating a dummy input of size (batch_size, channels, image_height, image_width)
    input_tensor = torch.ones(
        (batch_size, channels, image_height, image_width),
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
