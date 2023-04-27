"""Testing the loading/saving of internal resources."""

import os
import tempfile
from typing import List, Set

import importlib_resources

from hela.approximation.aliases import (
    Aliases,
    ModuleAliases,
    load_modules_aliases,
    save_modules_aliases,
)
from hela.approximation.module_to_approximate import (
    ModuleToApproximate,
)
from hela.approximation.pipeline_steps import (
    ApproximationStep,
    PipelineSteps,
    TrainingStepTrainerArgs,
    load_pipeline_steps,
    save_pipeline_steps,
)

ALIASES_FILE = str(
    importlib_resources.files("hela")
    / "resources"
    / "approximation"
    / "aliases.json"
)

# aliases.json


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

    assert isinstance(module_aliases, ModuleAliases)
    assert isinstance(module_aliases.name, str)
    assert isinstance(module_aliases.aliases, Set)
    assert isinstance(module_aliases.default_approximation_type, str)
    assert isinstance(module_aliases.dependencies, List)
    for dependency in module_aliases.dependencies:
        assert isinstance(dependency, ModuleToApproximate)

    assert module_aliases.name == tmp["name"]
    for alias in tmp["aliases"]:
        module_aliases.aliases.remove(alias)
    assert module_aliases.aliases == set()
    assert (
        module_aliases.default_approximation_type == tmp["default_approximation_type"]
    )


def test_load_modules_aliases():
    """Tests the loading of modules' aliases data structure from a file."""
    aliases = load_modules_aliases()

    # ASSERTS

    assert isinstance(aliases, Aliases)
    for item in aliases.aliases_list:
        assert isinstance(item, ModuleAliases)
        assert isinstance(item.name, str)
        assert isinstance(item.aliases, Set)


def test_save_modules_aliases():
    """Tests the saving of the modules' aliases data structure."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_modules_aliases(ALIASES_FILE, tmpdirname)

        # ASSERTS

        assert os.path.exists(os.path.join(tmpdirname, "aliases.json"))


def test_load_saved_aliases():
    """Tests the loading of the saved modules' aliases data structure."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_modules_aliases(ALIASES_FILE, tmpdirname)
        saved_aliases = Aliases.parse_file(os.path.join(tmpdirname, "aliases.json"))
        resource_aliases = Aliases.parse_file(ALIASES_FILE)

        # ASSERTS

        assert os.path.exists(os.path.join(tmpdirname, "aliases.json"))
        for idx in range(len(resource_aliases.aliases_list)):
            assert (
                saved_aliases.aliases_list[idx].name
                == resource_aliases.aliases_list[idx].name
            )
            assert (
                saved_aliases.aliases_list[idx].aliases
                == resource_aliases.aliases_list[idx].aliases
            )
            assert (
                saved_aliases.aliases_list[idx].default_approximation_type
                == resource_aliases.aliases_list[idx].default_approximation_type
            )


# pipeline_steps.json


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

    assert isinstance(pipeline_step, ApproximationStep)
    assert isinstance(pipeline_step.index, int)
    assert isinstance(pipeline_step.to_approximate, List)
    for item in pipeline_step.to_approximate:
        assert isinstance(item, ModuleToApproximate)
    assert isinstance(pipeline_step.training_args, TrainingStepTrainerArgs)


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

    assert isinstance(pipeline_steps, PipelineSteps)
    for pipeline_step in pipeline_steps.get_pipeline_step_list():
        assert isinstance(pipeline_step.index, int)
        assert isinstance(pipeline_step.to_approximate, List)
        for item in pipeline_step.to_approximate:
            assert isinstance(item, ModuleToApproximate)
        assert isinstance(pipeline_step.training_args, TrainingStepTrainerArgs)


def test_load_pipeline_steps():
    """Tests the loading of the pipeline steps data structure from a file."""
    pipeline_steps = load_pipeline_steps("./pipeline_steps.json")

    # ASSERTS

    assert isinstance(pipeline_steps, PipelineSteps)
    for step in pipeline_steps.get_pipeline_step_list():
        assert isinstance(step, ApproximationStep)
        assert isinstance(step.index, int)
        assert isinstance(step.to_approximate, List)
        assert isinstance(step.training_args, TrainingStepTrainerArgs)


def test_save_pipeline_steps():
    """Tests the saving of the pipeline steps data structure."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipeline_steps = load_pipeline_steps("./pipeline_steps.json")
        save_pipeline_steps(pipeline_steps, tmpdirname)

        # ASSERTS

        assert isinstance(pipeline_steps, PipelineSteps)
        assert os.path.exists(os.path.join(tmpdirname, "pipeline_steps.json"))


def test_load_saved_pipeline_steps():
    """Tests the loading of the saved pipeline steps data structure."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipeline_steps = load_pipeline_steps("./pipeline_steps.json")
        save_pipeline_steps(pipeline_steps, tmpdirname)
        saved_pipeline_steps = PipelineSteps.parse_file(
            os.path.join(tmpdirname, "pipeline_steps.json")
        )
        resource_pipeline_steps = PipelineSteps.parse_file("./pipeline_steps.json")

        # ASSERTS

        assert os.path.exists(os.path.join(tmpdirname, "pipeline_steps.json"))
        for idx in range(len(saved_pipeline_steps.pipeline_steps)):
            assert (
                saved_pipeline_steps.pipeline_steps[idx].index
                == resource_pipeline_steps.pipeline_steps[idx].index
            )
            assert (
                saved_pipeline_steps.pipeline_steps[idx].to_approximate
                == resource_pipeline_steps.pipeline_steps[idx].to_approximate
            )
            assert (
                saved_pipeline_steps.pipeline_steps[idx].training_args
                == resource_pipeline_steps.pipeline_steps[idx].training_args
            )
