"""Testing the vanilla transformer configuration."""

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from hela.models.vanilla_transformer.configuration import VanillaTransformerConfig


def test_configuration_init():
    """Tests the initialization of the configuation of a vanilla transformer."""

    # initializing (the default) configuration
    default_configuration = VanillaTransformerConfig()

    # ASSERTS

    assert isinstance(default_configuration, VanillaTransformerConfig)


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
