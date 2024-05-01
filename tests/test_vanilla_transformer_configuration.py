"""Testing the vanilla transformer configuration."""

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import pytest
import torch

from hela.models.vanilla_transformer.configuration import VanillaTransformerConfig

# defining the devices to run the tests on
DEVICE_LIST = ["cpu", "cuda"]


@pytest.mark.parametrize(
    "device",
    DEVICE_LIST,
    ids=[f"device: {device}" for device in DEVICE_LIST],
)
def test_configuration_init(device: str):
    """Tests the initialization of the configuation of a vanilla transformer."""

    # initializing (the default) configuration
    default_configuration = VanillaTransformerConfig(device=device)

    # ASSERTS

    assert isinstance(default_configuration, VanillaTransformerConfig)

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


def test_configuration_save():
    """Tests the saving of a vanilla transformer configuration."""

    with TemporaryDirectory() as temp_dir:
        # initializing the (default) vanilla transformer configuration
        default_configuration = VanillaTransformerConfig()

        # saving the latter configuration
        default_configuration.save_pretrained(temp_dir)

        # ASSERTS

        assert os.path.isdir(temp_dir)
        assert isinstance(default_configuration, VanillaTransformerConfig)


def test_configuration_load():
    """Tests the loading of a saved vanilla transformer configuration."""

    with TemporaryDirectory() as temp_dir:
        # initializing the (default) vanilla transformer configuration
        default_configuration = VanillaTransformerConfig()

        # saving the latter configuration
        default_configuration.save_pretrained(temp_dir)

        # loading the saved configuration (json)
        loaded_configuration = VanillaTransformerConfig.from_pretrained(temp_dir)

        # ASSERTS

        assert isinstance(default_configuration, VanillaTransformerConfig)
        assert isinstance(loaded_configuration, VanillaTransformerConfig)

        TestCase().assertDictEqual(
            default_configuration.to_dict(), loaded_configuration.to_dict()
        )
