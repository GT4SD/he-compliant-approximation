Approximation Resources
=======================

The resources used to determine the configuration of the approximation pipeline are validated using `Pydantic <https://github.com/pydantic/pydantic>`_
These include aliases for neural network modules, data structures for modules to be approximated, and the pipeline steps for approximation.

Module Aliases
--------------

The ``aliases.py`` module contains data structures for managing aliases of neural network modules and their approximation types.

- **ModuleAliases**: This class represents the mapping between a common module name and its corresponding PyTorch module names. It includes attributes for the module name, aliases, default approximation type, and dependencies.

- **Aliases**: This class manages a set of ``ModuleAliases``. It provides functionalities to access module aliases, dependencies, and default approximation types.

Module To Approximate
---------------------

The ``module_to_approximate.py`` module defines the structure for modules that need to be approximated.

- **ModuleToApproximate**: This class contains information about a module that needs to be approximated, including the module name, approximation type, and parameters. It also includes a validator to ensure that parameters for certain approximation types are correctly specified.

- **ToApproximate**: This class manages a set of ``ModuleToApproximate`` instances. It provides methods to access and modify this set, as well as to check if a module is included in the set.

Approximation Pipeline Steps
----------------------------

The ``pipeline_steps.py`` module outlines the structure for defining and managing the steps in an approximation pipeline.

- **TrainingStepTrainerArgs**: This class contains arguments for PyTorch Lightning trainers used in pipeline training steps. It includes various training configurations and early stopping parameters.

- **ApproximationStep**: This class represents a single step in the approximation pipeline, including the modules to approximate and the training arguments. It also includes a validator to check for conflicts between approximation parameters and training arguments.

- **PipelineSteps**: This class manages a list of ``ApproximationStep`` instances, representing the entire approximation pipeline. It provides methods to access and modify the pipeline steps, as well as to set flags related to early stopping.

Each of these modules plays a crucial role in the approximation process, allowing for flexible and configurable approximation of neural network modules.