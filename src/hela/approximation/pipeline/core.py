"""Implementation of the approximation pipeline."""

import json
import os
import re
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Dict, Optional, Set, Type

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from ...pytorch_lightning.models.approximations.core import LitApproximatedModel
from ..aliases import ALIASES_FILE, save_modules_aliases
from ..controller import ModelApproximationController
from ..module_to_approximate import ModuleToApproximate
from ..pipeline_steps import (
    ApproximationStep,
    TrainingStepTrainerArgs,
    load_pipeline_steps,
    save_pipeline_steps,
)


class ApproximationPipeline:
    """ApproximationPipeline core implementation."""

    def __init__(
        self,
        model: nn.Module,
        lightning_model_class: Type[pl.LightningModule],
        trainer_args: Dict[str, Any],
        pipeline_steps_path: str,
        experiment_ckpt: Optional[str] = None,
        modules_aliases_file: str = ALIASES_FILE,
        **kwargs,
    ) -> None:
        """Initializes the ApproximationPipeline.

        Args:
            model: model to be approximated in the pipeline.
            lightning_model_class: pytorch lightning class of the model to be approximated.
            trainer_args: arguments to be passed to the pytorch lightning trainer.
            pipeline_steps_path: path of the file containing the pipeline steps informations.
            experiment_ckpt: path of an existing experiment directory. Defaults to None.
            modules_aliases_file: path of the file containing the modules' aliases informations. Defaults to ALIASES_FILE.

        Raises:
            FileNotFoundError: the given experiment directory does not exist.
            FileNotFoundError: the given experiment directory does not exist.
            TypeError: the lightning_model_class is not a subclass of LitApproximatedModel
        """

        self.model = model

        self.experiment_ckpt = experiment_ckpt
        if self.experiment_ckpt is not None and not os.path.exists(
            self.experiment_ckpt
        ):
            raise FileNotFoundError(f"Experiment not found at {self.experiment_ckpt}.")

        # in case the pipeline should continue from a given experiment it should load the saved resources files
        if self.experiment_ckpt is not None:
            pipeline_steps_path = os.path.join(
                self.experiment_ckpt, "pipeline_steps.json"
            )
            modules_aliases_file = os.path.join(self.experiment_ckpt, "aliases.json")

        # loading the approximation pipeline steps informations
        self.pipeline_steps = load_pipeline_steps(file_path=pipeline_steps_path)

        self.trainer_args: Dict[str, Any]

        if self.experiment_ckpt is None:
            self.save_path = os.path.join(
                trainer_args["experiment_log_dir"],  # type: ignore
                trainer_args["experiment_name"],  # type: ignore
            )
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            save_pipeline_steps(self.pipeline_steps, self.save_path)
            save_modules_aliases(modules_aliases_file, self.save_path)
            self.trainer_args = trainer_args
            with open(
                os.path.join(self.save_path, "trainer_hyperparameters.json"), "w"
            ) as fp:
                json.dump(self.trainer_args, fp, indent=4, sort_keys=True)
        else:
            self.save_path = self.experiment_ckpt
            if not os.path.exists(self.save_path):
                raise FileNotFoundError(
                    f"The experiment at {self.save_path} does not exists."
                )
            with open(
                os.path.join(self.save_path, "trainer_hyperparameters.json"), "r"
            ) as fp:
                self.trainer_args = json.load(fp)

        # defining the approximation controller
        self.controller = ModelApproximationController(
            model=model,
            modules_aliases_file=modules_aliases_file,
        )
        self.lightning_model = lightning_model_class(
            model=model, controller=self.controller, model_args=trainer_args
        )
        if not isinstance(self.lightning_model, LitApproximatedModel):
            raise TypeError(
                f"Expected a lightning model class inherited from {LitApproximatedModel}. Obtained {lightning_model_class}."
            )

        # defining the variables to store the state of the pipeline
        self.current_step: int = 0
        self.current_to_approximate: Set[ModuleToApproximate] = set()
        self.current_training_args: TrainingStepTrainerArgs

        self.model_ckpt: Optional[str] = None

        self.step_ckpt: Optional[int] = None
        if self.experiment_ckpt is not None:
            self.step_ckpt = ApproximationPipeline.compute_steps_ckpt(
                self.experiment_ckpt
            )

        self.tensorboardLogger: TensorBoardLogger

    def save_model_structure(self) -> None:
        """Prints the model structure in a txt file inside the experiment folder."""
        file_path = os.path.join(self.save_path, "model_structure.txt")

        with open(file_path, "w") as fstruct:
            print(self.model, file=fstruct)

    def save_step_hyperparameters(self, step: ApproximationStep) -> None:
        """Saves the approximation step hyperparameters"""
        file_path = os.path.join(self.save_path, "step_hparams.json")
        with open(file_path, "w") as outfile:
            json.dump(json.loads(step.json()), outfile, indent=4)

    @staticmethod
    def add_pipeline_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds specific arguments for the pipeline to the parser.

        Args:
            parent_parser: argument parser to be updated.

        Returns:
            updated parser.
        """

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # approximation configuration arguments
        parser.add_argument(
            "--pipeline_steps_path",
            type=str,
            default=os.path.join(
                os.getcwd(), "./pipeline_steps/without_approximations.json"
            ),
        )
        parser.add_argument("--modules_aliases_file", type=str, default=ALIASES_FILE)
        parser.add_argument(
            "--experiment_log_dir",
            type=str,
            default=os.path.join(os.getcwd(), "pipeline_logs"),
        )
        parser.add_argument(
            "--experiment_name",
            type=str,
            default=f"experiment_{datetime.now().strftime('%d-%m-%y_%H:%M:%S')}",
        )
        parser.add_argument(
            "--experiment_ckpt",
            type=str,
            default=None,
            required=False,
        )

        return parser

    @staticmethod
    def compute_steps_ckpt(experiment_path: str) -> int:
        """Counts the number of steps directories inside the experiment directory.

        Args:
            experiment_path: experiment directory path.

        Raises:
            RuntimeError: the experiment directory does not exist.

        Returns:
            number of steps directories inside the experiment directory.
        """
        if not os.path.exists(experiment_path):
            raise RuntimeError(f"Does not exist an experiment path '{experiment_path}'")

        count = 0
        for file in os.listdir(experiment_path):
            if re.match(r"^step_\d+$", file) and os.path.isdir(
                os.path.join(experiment_path, file)
            ):
                count = count + 1

        return count
