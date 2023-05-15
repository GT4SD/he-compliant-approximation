"""Implementation of the testing approximation pipeline."""

import json
import os
from argparse import Namespace
from typing import Dict, Type, Union

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from ..aliases import ALIASES_FILE
from ..module_to_approximate import ToApproximate
from .core import ApproximationPipeline


class TestingPipeline(ApproximationPipeline):
    """TestingPipeline implementation."""

    def __init__(
        self,
        model: nn.Module,
        lightning_model_class: Type[pl.LightningModule],
        trainer_args: Dict[str, Union[float, int, str]],
        pipeline_steps_path: str,
        experiment_ckpt: str,
        test_dataloader: DataLoader,
        modules_aliases_file: str = ALIASES_FILE,
    ) -> None:
        """Initializes the TestingPipeline.

        Args:
            model: model to be approximated in the pipeline.
            lightning_model_class: pytorch lightning class of the model to be approximated.
            trainer_args: arguments to be passed to the pytorch lightning trainer.
            pipeline_steps_path: path of the file containing the pipeline steps informations.
            experiment_ckpt: path of an existing experiment directory.
            test_dataloader: dataloader of the test set.
            modules_aliases_file: path of the file containing the modules' aliases informations. Defaults to ALIASES_FILE.

        Raises:
            AttributeError: the testing pipeline needs an experiment checkpoint
            FileNotFoundError: the given experiment directory does not exist.
        """

        super().__init__(
            model,
            lightning_model_class,
            trainer_args,
            pipeline_steps_path,
            experiment_ckpt,
            modules_aliases_file,
        )

        if self.experiment_ckpt is None:
            raise AttributeError("Testing pipeline needs an experiment directory.")
        elif not os.path.exists(self.experiment_ckpt):
            raise FileNotFoundError(f"Experiment not found at {self.experiment_ckpt}.")

        self.test_dataloader = test_dataloader

    def test(self) -> None:
        """Tests the approximated model following the approximation pipeline of the given experiment checkpoint."""

        for step in self.pipeline_steps.get_pipeline_step_list():
            # updating the state of the pipeline
            self.current_step = step.index
            self.current_to_approximate = set(step.to_approximate)
            self.current_training_args = step.training_args

            # a simple fine-tuning step without approximation is identified with set()
            # if this is not a simple fine-tuning step approximate the model we should approximate the model
            if not self.current_to_approximate == set():
                # updating the model to be approximated by the controller
                self.controller.update_model(model=self.model)
                # updating the layers to be approximated for the current pipeline step
                self.controller.update_to_approximate(
                    to_approximate=ToApproximate(
                        **{"modules_set": self.current_to_approximate}
                    )
                )
                # approximating the model contained in the controller
                self.model = self.controller.get_approximated_model(pretrained=False)  # type: ignore
                # updating the model used by the lightning module, specifing the approximation controller
                self.lightning_model.update_model(  # type: ignore
                    new_model=self.model,
                    new_controller=self.controller,
                )

                # loading the model checkpoint from the last step of the training (if exists)
                self.load_model_checkpoint()

                # converting the approximated layer to their pretrained form
                self.model = self.controller.get_approximated_model(pretrained=True)  # type: ignore
                # updating the model used by the lightning module, specifing the approximation controller
                self.lightning_model.update_model(  # type: ignore
                    new_model=self.model,
                    new_controller=self.controller,
                )
            else:
                # loading the model checkpoint from the last step of the training (if exists)
                self.load_model_checkpoint()

        # defining the tensorboard logger for the testing
        self.tensorboard_logger = TensorBoardLogger(
            self.save_path,
            name=None,
            version="test",
        )

        # saving all the files in the test directory inside the experiment directory
        self.save_path = self.tensorboard_logger.log_dir
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_model_structure()

        # Testing
        ###############

        trainer = pl.Trainer.from_argparse_args(
            Namespace(**self.trainer_args),
            logger=self.tensorboard_logger,
        )

        trainer.test(
            self.lightning_model,
            dataloaders=self.test_dataloader,
        )

        ###############

        # saving the results of the test
        with open(os.path.join(self.save_path, "results.json"), "w") as file:
            json_dict = self.lightning_model.return_results_metrics(  # type: ignore
                support=len(self.test_dataloader.dataset)  # type: ignore
            )
            json.dump(json_dict, file, indent=4)

    def load_model_checkpoint(self) -> None:
        """Loads the model parameters from a checkpoint.

        Raises:
            ValueError: the checkpoint does not exists.
        """
        # for testing the model weights can be loaded at the end of the approximation process
        # in case self.step_ckpt > self.current_step the model may not be loaded
        if self.step_ckpt == self.current_step:
            # loading the model checkpoint from the previous step
            ckpt_dir_path = os.path.join(
                self.experiment_ckpt,  # type: ignore
                f"step_{self.current_step}",
                "checkpoints",
            )
            if os.path.exists(ckpt_dir_path):
                # attention: right now the loading of the ckpt assumes to have only 1 ckpt saved in the directory
                self.model_ckpt = os.path.join(
                    ckpt_dir_path, os.listdir(ckpt_dir_path)[0]
                )
            else:
                self.model_ckpt = None
            if self.model_ckpt is None:
                raise ValueError(
                    "Attempting to load a checkpoint that does not exists."
                )
            self.lightning_model = (
                self.lightning_model.__class__
            ).load_from_checkpoint(
                self.model_ckpt,
                model=self.model,
                controller=self.controller,
                model_args=self.trainer_args,
            )
            print(
                f"Restoring weights and hyperparameters states from the checkpoint path at {self.model_ckpt}"
            )
