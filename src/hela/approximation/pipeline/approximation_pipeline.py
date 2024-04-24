"""Implementation of the approximation pipeline."""

import json
import os
import re
from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.pruning import ModelPruning
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from ..aliases import ALIASES_FILE
from ..module_to_approximate import ToApproximate
from ..pipeline_steps import PipelineSteps, load_pipeline_steps, save_pipeline_steps
from .core import Pipeline


class ApproximationPipeline(Pipeline):
    """ApproximationPipeline implementation."""

    def __init__(
        self,
        model: nn.Module,
        lightning_model_class: Type[pl.LightningModule],
        lightning_model_args: Dict[str, Any],
        trainer_args: Dict[str, Any],
        pipeline_steps_path: str,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        modules_aliases_file: str = ALIASES_FILE,
        experiment_ckpt: Optional[str] = None,
    ) -> None:
        """Initializes the ApproximationPipeline

        Args:
            model: model to be approximated in the pipeline.
            lightning_model_class: pytorch lightning class of the model to be approximated.
            trainer_args: arguments to be passed to the pytorch lightning trainer.
            pipeline_steps_path: path of the file containing the pipeline steps informations.
            train_dataloader:  dataloader of the training set.
            val_dataloader:  dataloader of the validation set.
            modules_aliases_file: path of the file containing the modules' aliases informations. Defaults to ALIASES_FILE.
            experiment_ckpt: path of an existing experiment directory. Defaults to None.

        Raises:
            ValueError: in case a checkpoint experiment is not given the step_ckpt should be None.
        """

        super().__init__(
            model,
            lightning_model_class,
            lightning_model_args,
            trainer_args,
            pipeline_steps_path,
            experiment_ckpt,
            modules_aliases_file,
        )

        if self.experiment_ckpt is not None and self.step_ckpt is None:
            raise ValueError(
                f"Experiment checkpoint is None and step checkpoint is expected to be None. Instead step checkpoint is {self.step_ckpt}"
            )

        if self.experiment_ckpt is not None:
            print(f"Continuing training for experiment at {self.experiment_ckpt}")

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.not_trained = True

        self.experiment_dir = self.save_path

    def _update_save_path(self) -> None:
        """Updates the saving path to the directory for the current pipeline step."""
        if self.experiment_ckpt is None:
            # defining the logger for the new pipeline step
            self.tensorboard_logger = TensorBoardLogger(
                self.trainer_args["experiment_log_dir"],  # type: ignore
                name=self.trainer_args["experiment_name"],  # type: ignore
                version=f"step_{self.current_step}",
            )
        else:
            # defining the logger for the new pipeline step
            self.tensorboard_logger = TensorBoardLogger(
                self.experiment_ckpt,
                name=None,
                version=f"step_{self.current_step}",
            )
        self.save_path = self.tensorboard_logger.log_dir

        # creation of the directories path if not exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def load_model_checkpoint_after_fit(self) -> None:
        """Loads the model parameters from the current pipeline step checkpoint after the fit method being called.

        Raises:
            ValueError: the checkpoints folder does not exists.
            ValueError: the checkpoints folder is empty.
        """
        root_dir: str
        if self.experiment_ckpt is None:
            root_dir = self.tensorboard_logger.root_dir
        else:
            root_dir = self.experiment_ckpt

        ckpt_dir_path = os.path.join(
            root_dir,
            f"step_{self.current_step}",
            "checkpoints",
        )

        if not os.path.exists(ckpt_dir_path):
            raise ValueError(
                f"Attempting to load a checkpoint for step {self.current_step} but it does not exists the checkpoints folder. Something went wrong with the checkpoint saving during training."
            )
        elif len(os.listdir(ckpt_dir_path)) == 0:
            raise ValueError(
                f"Attempting to load a checkpoint for step {self.current_step} but the checkpoints folder is empty. Something went wrong with the checkpoint saving during training."
            )

        # NOTE: the checkpoint loaded is the first one in a descending order.
        self.model_ckpt = os.path.join(
            ckpt_dir_path, sorted(os.listdir(ckpt_dir_path), reverse=True)[-1]
        )

        self._load_model_state_dict_from_ckpt()

        print(
            f"Restoring weights states from the BEST model checkpoint at {self.model_ckpt}"
        )

    def _load_model_checkpoint_before_fit(self) -> None:
        """Loads the model parameters from the current pipeline step checkpoint before the fit method being called.

        Raises:
            FileNotFoundError: the checkpoints directory does not contain any checkpoint.
            FileNotFoundError: the checkpoints directory does not exist.
        """
        if (
            self.experiment_ckpt is not None
            and self.step_ckpt is not None
            and self.step_ckpt == self.current_step
        ):
            ckpt_dir_path = os.path.join(
                self.experiment_ckpt,
                f"step_{self.current_step}",
                "checkpoints",
            )
            if os.path.exists(ckpt_dir_path):
                # loading the saved ckpt from the latest epoch
                largest_epoch_path = None
                largest_epoch = -1

                for path in os.listdir(ckpt_dir_path):
                    match = re.search(r"epoch=(\d+)", path)
                    if match:
                        epoch = int(match.group(1))
                        if epoch > largest_epoch:
                            largest_epoch = epoch
                            largest_epoch_path = path
                if largest_epoch_path is None:
                    raise FileNotFoundError(
                        f"Does not exist any checkpoint in {ckpt_dir_path}."
                    )
                else:
                    self.model_ckpt = os.path.join(ckpt_dir_path, largest_epoch_path)
            else:
                raise FileNotFoundError(
                    f"Attempted to load a checkpoint for step {self.current_step} in experiment {self.experiment_ckpt} but it does not exists. Please check the folder of the experiment to continue the training."
                )

            self._load_model_state_dict_from_ckpt()

            print(f"Restoring weights states from the checkpoint at {self.model_ckpt}")

    def _get_default_model_checkpoint_callback(self) -> List[Callback]:
        """Defines the default model callbacks.

        Returns:
            default model checkpoint callback.
        """
        monitor = self.current_training_args.ckpt_monitor
        filename: Optional[str] = None
        if monitor is None:
            print(
                "WARNING: model checkpointing will be applied without monitoring a certain metric, saving the checkpoint for the last epoch. Its value can be modified specifing 'ckpt_mode' argument."
            )
        else:
            filename = f"{{epoch}}-{{{monitor}:.3f}}"
        mode = self.current_training_args.ckpt_mode

        callbacks: List = []
        callbacks.append(
            ModelCheckpoint(
                monitor=monitor,
                filename=filename,
                mode=mode,
            )
        )

        return callbacks

    def _get_early_stopping_callbacks(self) -> List[Callback]:
        """Defines the callbacks needed for early stopping.

        Returns:
            callbacks:
                - early stopping.
                - model checkpoint.
        """
        monitor = self.current_training_args.early_stopping_monitor
        patience = self.current_training_args.early_stopping_patience
        mode = self.current_training_args.early_stopping_mode

        callbacks: List = []
        callbacks.append(
            EarlyStopping(
                monitor=monitor,  # type: ignore
                patience=patience,
                mode=mode,
            )
        )
        callbacks.append(
            ModelCheckpoint(
                monitor=monitor,
                filename=f"{{epoch}}-{{{monitor}:.3f}}",
                mode=mode,
            )
        )

        return callbacks

    def _get_pruning_callback(self) -> List[Callback]:
        """Defines the callbacks needed for pruning.

        Returns:
            pruning callback.
        """
        callback: List = []
        callback.append(
            ModelPruning(
                pruning_fn=self.current_training_args.pruning_fn,
                parameters_to_prune=self.current_training_args.parameters_to_prune,
                use_global_unstructured=self.current_training_args.pruning_use_global_unstructured,
                amount=self.current_training_args.pruning_amount,
                apply_pruning=True,
                make_pruning_permanent=False,  # to avoid the pruning being lost in successive pipeline steps
                use_lottery_ticket_hypothesis=self.current_training_args.pruning_use_lottery_ticket_hypothesis,
                resample_parameters=self.current_training_args.pruning_resample_parameters,
                pruning_dim=self.current_training_args.pruning_dim,
                pruning_norm=self.current_training_args.pruning_norm,
                verbose=self.current_training_args.pruning_verbose,
                prune_on_train_epoch_end=self.current_training_args.prune_on_train_epoch_end,
            )
        )
        return callback

    def _get_skip_validation_model_checkpoint_callback(self) -> List[Callback]:
        """Defines the callbacks needed when the model is not validated.

        Returns:
            save-last model checkpoint callback.
        """
        print(
            f"Training for {self.trainer_args_for_current_step['max_epochs']} epochs WITHOUT validation."
        )
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-last",
            save_on_train_epoch_end=True,
        )
        return [checkpoint_callback]

    def fit(self) -> None:
        """Trains the approximated model following the given approximation pipeline steps."""

        for step in self.pipeline_steps.get_pipeline_step_list():
            # updating the state of the pipeline
            self.current_step = step.index
            self.current_to_approximate = set(step.to_approximate)
            self.current_training_args = step.training_args

            # updating the saving path for this experiment's step
            self._update_save_path()

            # a simple fine-tuning step without approximation is identified with set()
            # if this is not a simple fine-tuning step approximate the model we should approximate the model
            if not self.current_to_approximate == set():
                # updating the model to approximate by the controller
                self.controller.update_model(model=self.model)
                # updating the layers to be approximated for the current pipeline step
                self.controller.update_to_approximate(
                    to_approximate=ToApproximate(
                        **{"modules_set": self.current_to_approximate}
                    )
                )
                self.controller.update_save_path(save_path=self.save_path)
                # approximating the model contained in the controller
                self.model = self.controller.get_approximated_model(pretrained=False)  # type: ignore
                # updating the model used by the lightning module, specifing the approximation controller
                self.lightning_model.update_model(  # type: ignore
                    new_model=self.model,
                    new_controller=self.controller,
                )

            # the training step is done only if the training should not be resumed at a later step
            if (self.experiment_ckpt is None) or (
                self.experiment_ckpt is not None
                and self.step_ckpt is not None
                and self.step_ckpt <= self.current_step
            ):
                if (self.experiment_ckpt is None) or (
                    self.experiment_ckpt is not None
                    and self.step_ckpt is not None
                    and self.step_ckpt < self.current_step
                ):
                    # saving the model structure inside a file
                    self.save_model_structure()

                # the model's weights must be loaded after the approximation if the current training step was resumed and must reach the end
                if (
                    self.experiment_ckpt is not None
                    and self.step_ckpt is not None
                    and self.step_ckpt == self.current_step
                ):
                    # loading the model checkpoint from the last step of the training (if exists)
                    self._load_model_checkpoint_before_fit()

                # saving the current pipeline step hyperparameters in a file
                self.save_step_hyperparameters(step=step)

                trainer_fit_kwargs: Dict[str, Any] = {}

                # overwriting pytorch lightning trainer args for the current step with the ones specified for the pipeline step
                self.trainer_args_for_current_step = deepcopy(self.trainer_args)
                common_args = set(
                    self.trainer_args_for_current_step.keys()
                ).intersection(set(self.current_training_args.dict().keys()))
                for arg in common_args:
                    if getattr(self.current_training_args, arg) is not None:
                        self.trainer_args_for_current_step[arg] = getattr(
                            self.current_training_args, arg
                        )

                # adding trainer callbacks
                self.trainer_args_for_current_step["callbacks"] = []
                if self.current_training_args.skip_validation:
                    # adding save-last model checkpoint callback
                    self.trainer_args_for_current_step["callbacks"].extend(
                        self._get_skip_validation_model_checkpoint_callback()
                    )
                    self.trainer_args_for_current_step["limit_val_batches"] = 0
                elif self.current_training_args.early_stopping:
                    # adding early stopping and model checkpoint callbacks
                    self.trainer_args_for_current_step["callbacks"].extend(
                        self._get_early_stopping_callbacks()
                    )
                else:
                    # adding the default model checkpoint callback
                    self.trainer_args_for_current_step["callbacks"].extend(
                        self._get_default_model_checkpoint_callback()
                    )

                # adding pruning callback
                if self.current_training_args.pruning:
                    self.trainer_args_for_current_step["callbacks"].extend(
                        self._get_pruning_callback()
                    )

                # the trainer state must be loaded to continue the training for the current pipeline step
                if (
                    self.experiment_ckpt is not None
                    and self.step_ckpt is not None
                    and self.step_ckpt == self.current_step
                ):
                    trainer_fit_kwargs = {"ckpt_path": self.model_ckpt}
                else:
                    trainer_fit_kwargs = {"ckpt_path": None}

                # avoiding to resume the training of a pipeline step if the patience of early stopping was reached
                if (
                    self.experiment_ckpt is not None
                    and self.step_ckpt is not None
                    and self.step_ckpt == self.current_step
                    and self.current_training_args.early_stopping
                    and self.current_training_args.early_stopping_patience_reached
                ):
                    print(
                        f"Early stopping patience was reached for step {self.current_step}, hence the training will continue from the next step, if exists."
                    )
                    pass
                else:
                    trainer = pl.Trainer.from_argparse_args(
                        Namespace(**self.trainer_args_for_current_step),
                        logger=self.tensorboard_logger,
                    )

                    trainer.fit(
                        self.lightning_model,
                        train_dataloaders=self.train_dataloader,
                        val_dataloaders=self.val_dataloader,
                        **trainer_fit_kwargs,
                    )

                if (
                    self.current_training_args.early_stopping
                    and not self.current_training_args.early_stopping_patience_reached
                ):
                    pipeline_steps: PipelineSteps = load_pipeline_steps(
                        file_path=os.path.join(
                            os.path.dirname(self.save_path), "pipeline_steps.json"
                        )
                    )
                    pipeline_steps.set_early_stopping_patience_reached(
                        step_index=self.current_step,
                    )
                    save_pipeline_steps(
                        pipeline_steps, save_path=os.path.dirname(self.save_path)
                    )

                # loading the BEST model checkpoint obtained from the fit
                self.load_model_checkpoint_after_fit()

            # converting the approximated layer to their pretrained form
            if not self.current_to_approximate == set():
                self.model = self.controller.get_approximated_model(pretrained=True)  # type: ignore
                # updating the model used by the lightning module, specifying the approximation controller
                self.lightning_model.update_model(  # type: ignore
                    new_model=self.model,
                    new_controller=self.controller,
                )

        # setting flag to enable testing procedure
        self.not_trained = False

    def test(self) -> None:
        """Tests the approximated model."""

        assert not self.not_trained or (
            self.experiment_ckpt is not None
            and self.step_ckpt == len(self.pipeline_steps.get_pipeline_step_list())
        ), f"The model must be trained through all the pipeline before being tested. Trained for {self.step_ckpt} steps out of {len(self.pipeline_steps.get_pipeline_step_list())}."

        # if the fit has not been called but the experiment checkpoint contains all the pipeline steps
        if self.not_trained and (
            self.experiment_ckpt is not None
            and self.step_ckpt == len(self.pipeline_steps.get_pipeline_step_list())
        ):
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
                    # updating the model used by the lightning module, specifying the approximation controller
                    self.lightning_model.update_model(  # type: ignore
                        new_model=self.model,
                        new_controller=self.controller,
                    )

                    # converting the approximated layer to their pretrained form
                    self.model = self.controller.get_approximated_model(pretrained=True)  # type: ignore
                    # updating the model used by the lightning module, specifing the approximation controller
                    self.lightning_model.update_model(  # type: ignore
                        new_model=self.model,
                        new_controller=self.controller,
                    )

        # loading the model checkpoint from the last step of the training (if exists)
        self._load_model_testing_checkpoint()

        # defining the tensorboard logger for the testing
        self.tensorboard_logger = TensorBoardLogger(
            self.experiment_dir,
            name=None,
            version="test",
        )

        # saving all the files in the test directory inside the experiment directory
        save_path = self.tensorboard_logger.log_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_model_structure()

        # Testing
        ###############

        tmp_args = deepcopy(self.trainer_args)
        tmp_args["max_epochs"] = 1

        trainer = pl.Trainer.from_argparse_args(
            Namespace(**tmp_args),
            logger=self.tensorboard_logger,
        )

        trainer.test(
            self.lightning_model,
            dataloaders=self.test_dataloader,
        )

        ###############

        # saving the results of the test
        with open(os.path.join(save_path, "results.json"), "w") as file:
            json_dict = self.lightning_model.return_results_metrics(  # type: ignore
                support=len(self.test_dataloader.dataset)  # type: ignore
            )
            json.dump(json_dict, file, indent=4)

    def _load_model_testing_checkpoint(self) -> None:
        """Loads the model parameters from a checkpoint.
        For testing the model weights can be loaded at the end of the approximation process
        in case self.step_ckpt > self.current_step the model may not be loaded

        Raises:
            FileNotFoundError: the checkpoints directory does not contain any checkpoint.
            FileNotFoundError: the checkpoints directory does not exist.
        """

        assert not self.not_trained or (
            self.experiment_ckpt is not None
            and self.step_ckpt == len(self.pipeline_steps.get_pipeline_step_list())
        ), f"The model must be trained through all the pipeline before being tested. Trained for {self.step_ckpt} steps out of {len(self.pipeline_steps.get_pipeline_step_list())}."

        # loading the model checkpoint
        if self.experiment_ckpt is None:
            ckpt_dir_path = os.path.join(
                self.experiment_dir,  # type: ignore
                f"step_{len(self.pipeline_steps.get_pipeline_step_list())}",
                "checkpoints",
            )
        else:
            ckpt_dir_path = os.path.join(
                self.experiment_ckpt,  # type: ignore
                f"step_{self.step_ckpt}",
                "checkpoints",
            )

        if os.path.exists(ckpt_dir_path):
            # loading the saved ckpt from the latest epoch
            largest_epoch_path = None
            largest_epoch = -1

            for path in os.listdir(ckpt_dir_path):
                match = re.search(r"epoch=(\d+)", path)
                if match:
                    epoch = int(match.group(1))
                    if epoch > largest_epoch:
                        largest_epoch = epoch
                        largest_epoch_path = path

            if largest_epoch_path is None:
                raise FileNotFoundError(
                    f"Does not exist any checkpoint in {ckpt_dir_path}."
                )
            else:
                self.model_ckpt = os.path.join(ckpt_dir_path, largest_epoch_path)
        else:
            self.model_ckpt = None

        if self.model_ckpt is None:
            raise FileNotFoundError(
                f"The checkpoint directory at {ckpt_dir_path} does not exist."
            )

        self._load_model_state_dict_from_ckpt()

        print(
            f"Restoring weights and hyperparameters states from the checkpoint path at {self.model_ckpt}"
        )

    def _load_model_state_dict_from_ckpt(self) -> None:
        """Loads the model parameters from a checkpoint."""
        checkpoint = torch.load(self.model_ckpt)
        model_weights = checkpoint["state_dict"]

        # updating the keys by dropping `model.`
        for key in list(model_weights):
            model_weights[key.replace("model.", "")] = model_weights.pop(key)

        # self.model.load_state_dict(model_weights)
        for key in list(model_weights):
            value = self.model
            for attr in key.split("."):
                value = getattr(value, attr)
            value.data = model_weights.pop(key)
