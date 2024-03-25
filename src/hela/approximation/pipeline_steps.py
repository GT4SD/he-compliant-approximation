"""Approximation pipeline data structures."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, List, Optional, Set, Type

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from pydantic import BaseModel, model_validator

from .module_to_approximate import ModuleToApproximate


class TrainingStepTrainerArgs(BaseModel):
    """Contains the pytorch lightning trainer's arguments for a pipeline training step.
    Some arguments correspond to the one defined in pytorch lightning.
    others instead are used to specify some behaviours for the training pipeline step (e.g. apply early stopping).

    Attributes:
        max_epochs: maximum number of training epochs. Defaults to None.
        check_val_every_n_epoch: number of epochs between each validation. Defaults to 1.
        skip_validation: whether to skip validation. Defaults to False.
        ckpt_monitor: metric to monitor to select the best model when early stopping is not applied. Defaults to None.
        ckpt_mode: modality to select the best model when early stopping is not applied. Defaults to "min".
        early_stopping: whether to apply early stopping. Defaults to False.
        early_stopping_monitor: metric to monitor for early stopping. Defaults to None.
        early_stopping_mode: modality to select the best model. Defaults to "min".
        early_stopping_patience: number of validation without improvement to wait before stopping. Defaults to 3.
        early_stopping_patience_reached: whether the training step was stopped due to the early stopping patience being reached. Defaults to False.
    """

    max_epochs: Optional[int] = None
    check_val_every_n_epoch: int = 1

    skip_validation: bool = False

    ckpt_monitor: Optional[str] = None
    ckpt_mode: Literal["min", "max"] = "min"

    early_stopping: bool = False
    early_stopping_monitor: Optional[str] = None
    early_stopping_mode: Literal["min", "max"] = "min"
    early_stopping_patience: int = 3
    early_stopping_patience_reached: bool = False

    pruning: bool = False
    pruning_fn: Literal[
        "ln_structured", "l1_unstructured", "random_structured", "random_unstructured"
    ] = "random_unstructured"
    parameters_to_prune: Optional[List[str]] = None
    pruning_amount: float = 0.5
    pruning_use_global_unstructured: bool = True
    pruning_use_lottery_ticket_hypothesis: bool = True
    pruning_resample_parameters: bool = False
    pruning_dim: Optional[int] = None
    pruning_norm: Literal["L1", "L2"] = "L1"
    pruning_verbose: int = 0
    prune_on_train_epoch_end: bool = True

    @model_validator(mode="before")
    @classmethod
    def check_early_stopping_monitor_omitted(
        cls: Type[TrainingStepTrainerArgs], values: Any
    ) -> Any:
        """Validates the values related to early stopping.

        Args:
            cls: pytorch lightning trainer's arguments for a pipeline training step.
            values: trainer's arguments values.

        Raises:
            ValueError: the metric to be monitored for early stopping must be specified.

        Returns:
            trainer's arguments values.
        """
        early_stopping, monitor = values.get("early_stopping"), values.get(
            "early_stopping_monitor"
        )
        if early_stopping and (monitor is None):
            raise ValueError(
                "Early stopping was requested but early_stopping_monitor wasn't specified."
            )
        return values


class ApproximationStep(BaseModel):
    """Contains the information for the approximation step.

    Attributes:
        index: numerical index of the approximation step.
        to_approximate: set of module approximations.
        training_args: pytorch lightining trainer's arguments.
    """

    index: int
    to_approximate: Set[ModuleToApproximate]
    training_args: TrainingStepTrainerArgs

    @model_validator(mode="before")
    @classmethod
    def check_batch_normalization_with_TID_parameters(
        cls: Type[ApproximationStep], values: Any
    ) -> Any:
        """Validates the training arguments for the approximation of LayerNorm with Regularized BatchNorm.

        Args:
            cls: approximation step.
            values: approximation step values.

        Raises:
            ValueError: .

        Returns:
            approximation step values.
        """
        to_approximate, training_args = values.get("to_approximate"), values.get(
            "training_args"
        )
        for item in to_approximate:
            # checking whether the batch normalization parameters are in conflict with the training arguments
            if item["approximation_type"] == "batchnorm":
                if (
                    item["parameters"].get("regularized_BN", None) is not None
                    and training_args.skip_validation
                ):
                    raise ValueError(
                        "Regularized Batch Normalization should run without the 'skip_validation' flag."
                    )
        return values

    def __lt__(self, other: ApproximationStep) -> bool:
        """Compares two approximation steps objects order using their index attribute.

        Args:
            other: approximation step to be compared to.

        Returns:
            whether the approximation step is lower than the other.
        """
        return self.index < other.index

    def __le__(self, other: ApproximationStep) -> bool:
        """Compares two approximation steps objects order using their index attribute.

        Args:
            other: approximation step to be compared to.

        Returns:
            whether the approximation step is lower or equal than the other.
        """
        return self.index <= other.index

    def __gt__(self, other: ApproximationStep) -> bool:
        """Compares two approximation steps objects order using their index attribute.

        Args:
            other: approximation step to be compared to.

        Returns:
            whether the approximation step is greater than the other.
        """
        return self.index > other.index

    def __ge__(self, other: ApproximationStep) -> bool:
        """Compares two approximation steps objects order using their index attribute.

        Args:
            other: approximation step to be compared to.

        Returns:
            whether the approximation step is greater or equal than the other.
        """
        return self.index >= other.index

    def __hash__(self) -> int:
        """Fetches the hash value of the approximation step object.

        Returns:
            hash value of the approximation step object.
        """
        return hash(self.index)


class PipelineSteps(BaseModel):
    """Contains the list of approximation steps for a pipeline.

    Attributes:
        pipeline_steps: list of approximation steps.
    """

    pipeline_steps: List[ApproximationStep]

    def __init__(self, *args, **data):
        """Initializes the data structure."""
        super().__init__(*args, **data)
        # sorting the pipeline's approximation steps based on their index
        self.pipeline_steps.sort()
        # adjusting approximation steps indexes
        for index, step in enumerate(self.pipeline_steps):
            step.index = index + 1

    def get_pipeline_step_list(self) -> List[ApproximationStep]:
        """Gets the list of approximation steps.

        Returns:
            list of approximation steps.
        """
        return self.pipeline_steps

    def set_early_stopping_patience_reached(self, step_index: int) -> None:
        """Sets the early_stopping_patience_reached flag for a certain approximation step.

        Args:
            step_index: step for which the early stopping patience was reached.
        """
        setattr(
            self.pipeline_steps[step_index - 1].training_args,
            "early_stopping_patience_reached",
            True,
        )


def load_pipeline_steps(file_path: str) -> PipelineSteps:
    """Loads the pipeline approximation steps from a file.

    Args:
        file_path: path of the file to be parsed.

    Returns:
        pipeline steps data structure.
    """
    with open(file_path) as pipeline_steps_file:
        return PipelineSteps.model_validate(json.load(pipeline_steps_file))


def save_pipeline_steps(pipeline_steps: PipelineSteps, save_path: str) -> None:
    """Saves the pipeline approximation steps data structure in a json file.

    Args:
        pipeline_steps: pipeline approximation steps data structure.
        save_path: path of the saving directory.
    """
    save_path = os.path.join(save_path, "pipeline_steps.json")

    with open(save_path, "w") as outfile:
        json.dump(json.loads(pipeline_steps.model_dump_json()), outfile, indent=4)
