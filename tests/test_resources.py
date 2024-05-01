"""Testing the loading/saving of internal resources."""

import json
import os
import tempfile
from typing import Set

import importlib_resources

from hela.approximation.aliases import (
    Aliases,
    ModuleAliases,
    load_modules_aliases,
    save_modules_aliases,
)
from hela.approximation.module_to_approximate import ModuleToApproximate
from hela.approximation.pipeline_steps import (
    ApproximationStep,
    PipelineSteps,
    TrainingStepTrainerArgs,
    load_pipeline_steps,
    save_pipeline_steps,
)

ALIASES_FILE = str(
    importlib_resources.files("hela") / "resources" / "approximation" / "aliases.json"
)

PIPELINE_STEPS_FILE = "./pipeline_steps/vanilla_transformer/without_approximations.json"

################
# aliases.json #
################


def test_alias_mapping_init():
    """Tests the initialization of a modules' aliases data structure."""
    tmp = {
        "name": "tmp_module",
        "aliases": ["tmp_dep_1", "tmp_dep_2", "tmp_dep_3"],
        "default_approximation_type": "tmp_type",
        "dependencies": [
            {
                "module": "tmp_module",
                "approximation_type": "tmp_type",
                "parameters": {},
            }
        ],
    }

    module_aliases = ModuleAliases(**tmp)

    # ASSERTS

    assert isinstance(
        module_aliases, ModuleAliases
    ), "The module aliases should be an instance of ModuleAliases."
    assert isinstance(module_aliases.name, str), "The module name should be a string."
    assert isinstance(
        module_aliases.aliases, Set
    ), "The module's aliases should be a Set."
    assert isinstance(
        module_aliases.default_approximation_type, str
    ), "The module's default approximation type should be a string."
    assert isinstance(
        module_aliases.dependencies, Set
    ), "The module's dependencies should be a Set."
    for dependency in module_aliases.dependencies:
        assert isinstance(
            dependency, ModuleToApproximate
        ), "Each dependency should be an instance of ModuleToApproximate."

    assert (
        module_aliases.name == tmp["name"]
    ), "The module name should match is not initialized correctly."
    for alias in tmp["aliases"]:
        module_aliases.aliases.remove(alias)
    assert (
        module_aliases.aliases == set()
    ), "The module's aliases should be empty after removing all the aliases."
    assert (
        module_aliases.default_approximation_type == tmp["default_approximation_type"]
    ), "The module's default approximation type is not initialized correctly."


def test_load_modules_aliases():
    """Tests the loading of modules' aliases data structure from a file."""
    aliases = load_modules_aliases()

    # ASSERTS

    assert isinstance(
        aliases, Aliases
    ), "Loaded aliases should be an instance of Aliases."
    for item in aliases.aliases_set:
        assert isinstance(
            item, ModuleAliases
        ), "Each loaded module aliases should be an instance of ModuleAliases."
        assert isinstance(
            item.name, str
        ), "Each loaded module's name should be a string."
        assert isinstance(
            item.aliases, Set
        ), "Each loaded module's aliases should be a Set."


def test_save_modules_aliases():
    """Tests the saving of the modules' aliases data structure."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_modules_aliases(ALIASES_FILE, tmpdirname)

        # ASSERTS

        assert os.path.exists(
            os.path.join(tmpdirname, "aliases.json")
        ), "The aliases JSON file was not saved to the temporary directory."


def test_load_saved_aliases():
    """Tests the loading of the saved modules' aliases data structure."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_modules_aliases(ALIASES_FILE, tmpdirname)
        with open(os.path.join(tmpdirname, "aliases.json")) as saved_aliases_file:
            saved_aliases = Aliases.model_validate(json.load(saved_aliases_file))
        with open(ALIASES_FILE) as resource_aliases_file:
            resource_aliases = Aliases.model_validate(json.load(resource_aliases_file))

        # ASSERTS

        assert os.path.exists(
            os.path.join(tmpdirname, "aliases.json")
        ), "The aliases JSON file was not saved to the temporary directory."
        resource_aliases_list = sorted(list(resource_aliases.aliases_set))
        saved_aliases_list = sorted(list(saved_aliases.aliases_set))
        for idx in range(len(resource_aliases_list)):
            assert (
                saved_aliases_list[idx].name == resource_aliases_list[idx].name
            ), "The saved modules' names should match the names in the loaded resource."
            assert (
                saved_aliases_list[idx].aliases == resource_aliases_list[idx].aliases
            ), "The saved modules' aliases should match the aliases in the loaded resource."
            assert (
                saved_aliases_list[idx].default_approximation_type
                == resource_aliases_list[idx].default_approximation_type
            ), "The default approximation type of saved aliases should match the default approximation type in the loaded resource."


#######################
# pipeline_steps.json #
#######################


def test_approximation_step_init():
    """Tests the initialization of an approximation step data structure."""
    tmp = {
        "index": 1,
        "to_approximate": [
            {
                "module": "relu",
                "approximation_type": "quadratic",
                "parameters": {},
            },
        ],
        "training_args": {},
    }

    pipeline_step = ApproximationStep(**tmp)

    # ASSERTS

    assert isinstance(
        pipeline_step, ApproximationStep
    ), "The pipeline step should be an instance of ApproximationStep."
    assert isinstance(
        pipeline_step.index, int
    ), "The pipeline step's index should be an instance of int."
    assert isinstance(
        pipeline_step.to_approximate, Set
    ), "The pipeline step's modules to approximate should be an instance of Set."
    for item in pipeline_step.to_approximate:
        assert isinstance(
            item, ModuleToApproximate
        ), "Each module to be approximated in the step should be an instance of ModuleToApproximate."
    assert isinstance(
        pipeline_step.training_args, TrainingStepTrainerArgs
    ), "The training arguments for the step should be an instance of TrainingStepTrainerArgs."


def test_pipeline_steps_init():
    """Tests the initialization of a pipeline steps data structure."""
    tmp = {
        "pipeline_steps": [
            {
                "index": 1,
                "to_approximate": [
                    {
                        "module": "relu",
                        "approximation_type": "quadratic",
                        "parameters": {},
                    },
                ],
                "training_args": {},
            }
        ]
    }

    pipeline_steps = PipelineSteps(**tmp)

    # ASSERTS

    assert isinstance(
        pipeline_steps, PipelineSteps
    ), "Initialized pipeline steps should be an instance of PipelineSteps."
    for pipeline_step in pipeline_steps.get_pipeline_step_list():
        assert isinstance(
            pipeline_step.index, int
        ), "Pipeline step's index should be an instance of int."
        assert isinstance(
            pipeline_step.to_approximate, Set
        ), "Pipeline step's modules to approximate should be an instance of Set."
        for item in pipeline_step.to_approximate:
            assert isinstance(
                item, ModuleToApproximate
            ), "Each module to be approximated in the step should be an instance of ModuleToApproximate."
        assert isinstance(
            pipeline_step.training_args, TrainingStepTrainerArgs
        ), "The training arguments for the step should be an instance of TrainingStepTrainerArgs."


def test_load_pipeline_steps():
    """Tests the loading of the pipeline steps data structure from a file."""
    pipeline_steps = load_pipeline_steps(PIPELINE_STEPS_FILE)

    # ASSERTS

    assert isinstance(
        pipeline_steps, PipelineSteps
    ), "Loaded pipeline_steps should be an instance of PipelineSteps."
    for step in pipeline_steps.get_pipeline_step_list():
        assert isinstance(
            step, ApproximationStep
        ), f"Each step should be an instance of ApproximationStep, found {type(step)}."
        assert isinstance(
            step.index, int
        ), f"Step index should be of type int, found {type(step.index)}."
        assert isinstance(
            step.to_approximate, Set
        ), f"The pipeline step's to_approximate attribute should be of type Set, found {type(step.to_approximate)}."
        assert isinstance(
            step.training_args, TrainingStepTrainerArgs
        ), f"The pipeline step's training_args attribute should be an instance of TrainingStepTrainerArgs, found {type(step.training_args)}."


def test_save_pipeline_steps():
    """Tests the saving of the pipeline steps data structure."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipeline_steps = load_pipeline_steps(PIPELINE_STEPS_FILE)
        save_pipeline_steps(pipeline_steps, tmpdirname)

        # ASSERTS

        assert isinstance(
            pipeline_steps, PipelineSteps
        ), "Loaded pipeline steps is not an instance of PipelineSteps."
        assert os.path.exists(
            os.path.join(tmpdirname, "pipeline_steps.json")
        ), "Pipeline steps JSON file was not saved."


def test_load_saved_pipeline_steps():
    """Tests the loading of the saved pipeline steps data structure."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipeline_steps = load_pipeline_steps(PIPELINE_STEPS_FILE)
        save_pipeline_steps(pipeline_steps, tmpdirname)
        with open(
            os.path.join(tmpdirname, "pipeline_steps.json")
        ) as saved_pipeline_steps_file:
            saved_pipeline_steps = PipelineSteps.model_validate(
                json.load(saved_pipeline_steps_file)
            )
        with open(PIPELINE_STEPS_FILE) as resource_pipeline_steps_file:
            resource_pipeline_steps = PipelineSteps.model_validate(
                json.load(resource_pipeline_steps_file)
            )

        # ASSERTS

        assert os.path.exists(
            os.path.join(tmpdirname, "pipeline_steps.json")
        ), "Pipeline steps JSON file was not saved to the temporary directory."
        for idx in range(len(saved_pipeline_steps.pipeline_steps)):
            assert (
                saved_pipeline_steps.pipeline_steps[idx].index
                == resource_pipeline_steps.pipeline_steps[idx].index
            ), f"Discrepancy in index at step {idx}: Expected {resource_pipeline_steps.pipeline_steps[idx].index}, found {saved_pipeline_steps.pipeline_steps[idx].index}."
            assert (
                saved_pipeline_steps.pipeline_steps[idx].to_approximate
                == resource_pipeline_steps.pipeline_steps[idx].to_approximate
            ), f"Difference in modules to approximate at step {idx}: expected {resource_pipeline_steps.pipeline_steps[idx].to_approximate}, found {saved_pipeline_steps.pipeline_steps[idx].to_approximate}."
            assert (
                saved_pipeline_steps.pipeline_steps[idx].training_args
                == resource_pipeline_steps.pipeline_steps[idx].training_args
            ), f"Disparity in training arguments at step {idx}: expected {resource_pipeline_steps.pipeline_steps[idx].training_args}, found {saved_pipeline_steps.pipeline_steps[idx].training_args}."
