import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime
from typing import Callable
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from .experiment import Experiment, ExperimentResult
from .fit_experiment_result_factory import FitExperimentResultFactory
from ..data.modules import DataModule
from ..evaluation.evaluators import Evaluator, TrainEvaluator
from ..train import callbacks
from ..train.callbacks import Callback
from ..train.fit_output import FitOutput
from ..train.trainer import Trainer
from ..utils import logging as logging_utils
from ..utils import module as module_utils
from ..utils.parallel import DataParallelPassthrough


class ScoreInfo:

    def __init__(self, metric_name: str, is_train_metric: bool, largest: bool, return_best_score: bool):
        self.metric_name = metric_name
        self.is_train_metric = is_train_metric
        self.largest = largest
        self.return_best_score = return_best_score


class FitExperimentBase(Experiment, ABC):
    """
    Base abstract class for implementing an Experiment.
    """

    LOGGING_CALLBACK_NAME = "logging"
    METRICS_PLOTTER_CALLBACK_NAME = "metrics_plotter"
    CHECKPOINT_CALLBACK_NAME = "checkpoint"
    EARLY_STOPPING_CALLBACK_NAME = "early_stop"
    TENSORBOARD_CALLBACK_NAME = "tensorboard"
    WANDB_CALLBACK_NAME = "wandb"

    @staticmethod
    def add_experiment_base_specific_args(parser):
        """
        Adds supported run arguments to the given parser.
        """
        parser.add_argument("--experiment_name", type=str, default="exp", help="name of current experiment")

        parser.add_argument("--random_seed", type=int, default=-1, help="initial random seed")
        parser.add_argument("--gpu_ids", type=int, nargs="+", default=[], help="ids of cuda gpus to use")
        parser.add_argument("--trainer_checkpoint", type=str, default="", help="path to trainer checkpoint to continue training from")
        parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
        parser.add_argument("--validate_every", type=int, default=1, help="run validation every this number of epochs")

        parser.add_argument("--outputs_dir", type=str, default="outputs/exps", help="directory to create the experiment output folder in")
        parser.add_argument("--disable_console_log", action='store_true', help="do not log progress to console")
        parser.add_argument("--save_logs", action='store_true', help="save logs to file")
        parser.add_argument("--train_batch_log_interval", type=int, default=50, help="log train batch progress every this number of batches")
        parser.add_argument("--epoch_log_interval", type=int, default=1, help="log epoch progress every this number of epochs")
        parser.add_argument("--save_metric_plots", action='store_true', help="save plots for scalar metric values using matplotlib")
        parser.add_argument("--save_every_num_val", type=int, default=1, help="save checkpoints and plots every this number of validations")
        parser.add_argument("--use_tensorboard", action='store_true', help="write metric values to tensorboard")
        parser.add_argument("--use_wandb", action='store_true', help="activates wandb callback for tracking experiment")
        parser.add_argument("--wandb_project_name", type=str, default="default", help="wandb project name used to group different experiments")
        parser.add_argument("--wandb_entity_name", type=str, default="",
                            help="wandb entity (user/team) to report results under. Default is logged in user")
        parser.add_argument("--wandb_resume_id", type=str, default="", help="path of existing wandb experiment to resume from")
        parser.add_argument("--wandb_track_model", type=str, default=None, help="enable wandb tracking model information. "
                                                                                "Supports: 'gradients', 'parameters', 'all' (see wandb docs)")
        parser.add_argument("--wandb_exclude_files", nargs="+", type=str, default=[f"{callbacks.MetricsPlotter.DEFAULT_FOLDER_NAME}/**"],
                            help="sequence of file paths or glob to ignore when tracking files created in the experiment directory. "
                                 "Default is to ignore plots.")

        parser.add_argument("--score_metric_name", type=str, default="",
                            help="name of the metric to return as the experiment score. Also used for saving "
                                 "checkpoints and early stopping. If empty, will checkpoint according to epochs")
        parser.add_argument("--is_train_metric", action='store_true', help="experiment score is a train metric (default is validation metric)")
        parser.add_argument("--score_largest", type=bool, default=True, help="if true then larger score is better, if false then lower is better")
        parser.add_argument("--return_best_score", action='store_true',
                            help="if true then returns best score as experiment result (default is last score)")

        parser.add_argument("--save_checkpoints", action='store_true', help="store checkpoints of the trainer during training")
        parser.add_argument("--num_checkpoints", type=int, default=1, help="number of checkpoints to store")
        parser.add_argument("--save_checkpoints_by_score", action='store_true', help="if True, will save checkpoints according to score, otherwise "
                                                                                     "(default) saves the last checkpoints")
        parser.add_argument("--early_stop", action='store_true', help="use early stopping for training")
        parser.add_argument("--early_stop_min_delta", type=float, default=0,
                            help="minimum change to be considered an improvement in an epoch for early stopping")
        parser.add_argument("--early_stop_patience", type=int, default=0,
                            help="number of checks with no improvement after which training will be stopped.")
        parser.add_argument("--early_stop_cooldown", type=int, default=0,
                            help="number of checks with no improvement after which training will be stopped.")
        parser.add_argument("--early_stop_restore_best_weights", action='store_true',
                            help="Restore model weights from the epoch with the best score.")

    def validate_config(self, config: dict):
        """
        Verifies that the given configuration is valid (e.g. contains necessary fields).
        :param config: configuration dictionary.
        """
        if "epochs" not in config:
            raise ValueError("Missing 'epochs' parameter in the given fit parameters.")

    def initialize(self, config: dict, state: dict):
        """
        Runs any necessary initialization code. For example, initializes the random seed.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        """
        if "random_seed" in config:
            random_seed = config["random_seed"]
            self.__set_initial_random_seed(random_seed)

        gpu_ids = config["gpu_ids"] if "gpu_ids" in config else []
        main_gpu_id = gpu_ids[0] if gpu_ids else 0
        disable_gpu = len(gpu_ids) == 0
        device = module_utils.get_device(disable_gpu, main_gpu_id)
        state["gpu_ids"] = gpu_ids
        state["device"] = device

        experiment_start_time = datetime.utcnow()
        experiment_start_time_str = experiment_start_time.strftime("%Y_%m_%d-%H_%M_%S")
        outputs_dir = config["outputs_dir"]
        experiment_name = config["experiment_name"]
        experiment_dir = os.path.join(outputs_dir, f"{experiment_name}_{experiment_start_time_str}")

        state["experiment_start_time"] = experiment_start_time
        state["experiment_dir"] = experiment_dir
        state["logger"] = logging_utils.create_logger(console_logging=not config["disable_console_log"],
                                                      file_logging=config["save_logs"],
                                                      log_dir=state["experiment_dir"],
                                                      log_file_name_prefix=experiment_name,
                                                      timestamp=experiment_start_time)
        if config["save_logs"]:
            self.__save_configuration_files(experiment_dir=experiment_dir, config=config, context=state["context"])

    def __save_configuration_files(self, experiment_dir: str, config: dict, context: dict = None,
                                   config_file_name: str = "config.json", context_file_name: str = "context.json"):
        config_file_path = os.path.join(experiment_dir, config_file_name) if config is not None else ""
        if config_file_path:
            with open(config_file_path, "w") as f:
                json.dump(config, f, indent=2)

        context_file_path = os.path.join(experiment_dir, context_file_name) if context is not None else ""
        if context_file_path:
            with open(context_file_path, "w") as f:
                json.dump(context, f, indent=2)

    def __set_initial_random_seed(self, random_seed: int):
        if random_seed != -1:
            np.random.seed(random_seed)
            torch.random.manual_seed(random_seed)
            random.seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

    @abstractmethod
    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        """
        Creates the DataModule.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        :param logger: experiment logger.
        :return: the DataModule.
        """
        raise NotImplementedError

    @abstractmethod
    def create_model(self, datamodule: DataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        """
        Creates the model.
        :param datamodule: the datamodule.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        :param logger: experiment logger.
        :return: The model.
        """
        raise NotImplementedError

    def create_train_dataloader(self, datamodule: DataModule, config: dict, state: dict, logger: logging.Logger) -> torch.utils.data.DataLoader:
        """
        Creates the train dataloader.
        :param datamodule: the datamodule.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        :param logger: experiment logger.
        :return: Train dataloader.
        """
        return datamodule.train_dataloader()

    @abstractmethod
    def create_train_and_validation_evaluators(self, model: nn.Module, datamodule: DataModule, device, config: dict, state: dict,
                                               logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        """
        Creates the train and validation evaluators.
        :param model: PyTorch model.
        :param datamodule: the datamodule.
        :param device: device to use for evaluation.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        :param logger: experiment logger.
        :return: Train evaluator, Validation evaluator.
        """
        raise NotImplementedError

    @abstractmethod
    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        """
        :return: default information for the metric to use as a score for the experiment result, checkpointing,
        and early stopping (in case none given explicitly)
        """
        raise NotImplementedError

    def __get_score_info(self, config: dict, state: dict) -> ScoreInfo:
        if not config["score_metric_name"]:
            return self.get_default_score_info(config, state)

        return ScoreInfo(config["score_metric_name"], config["is_train_metric"], config["score_largest"], config["return_best_score"])

    def __create_score_fn(self, score_info: ScoreInfo) -> Callable[[Trainer], float]:
        def score_fn(trainer: Trainer):
            evaluator = trainer.train_evaluator if score_info.is_train_metric else trainer.val_evaluator
            return evaluator.get_tracked_values()[score_info.metric_name].current_value

        return score_fn

    def create_trainer_callback(self, model: nn.Module, datamodule: DataModule, config: dict, state: dict, logger: logging.Logger) -> Callback:
        """
        Creates a callback for the trainer. Can be overridden to customize callbacks.
        :param model: PyTorch model.
        :param datamodule: the datamodule.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        :param logger: experiment logger.
        :return: Callback to be called by the trainer during fitting.
        """
        additional_metadata = self.create_additional_metadata_to_log(model, datamodule, config, state, logger)
        logging_callback = self.__create_logging_callback(logger, config, state, additional_metadata=additional_metadata)
        callbacks_dict = OrderedDict({FitExperimentBase.LOGGING_CALLBACK_NAME: logging_callback})

        save_interval = config["save_every_num_val"] * config["validate_every"]
        if config["save_metric_plots"]:
            callbacks_dict[FitExperimentBase.METRICS_PLOTTER_CALLBACK_NAME] = callbacks.MetricsPlotter(state["experiment_dir"],
                                                                                                       create_plots_interval=save_interval)
        score_info = self.__get_score_info(config, state)
        score_fn = self.__create_score_fn(score_info)
        if config["save_checkpoints"]:
            checkpoints_score_fn = score_fn if config["save_checkpoints_by_score"] else None
            callbacks_dict[FitExperimentBase.CHECKPOINT_CALLBACK_NAME] = callbacks.Checkpoint(state["experiment_dir"],
                                                                                              save_interval=save_interval,
                                                                                              n_saved=config["num_checkpoints"],
                                                                                              score_function=checkpoints_score_fn,
                                                                                              score_name=config["score_metric_name"],
                                                                                              largest=config["score_largest"])

        if config["use_tensorboard"]:
            self.__add_tensorboard_callback(callbacks_dict, config, state)

        if config["use_wandb"]:
            self.__add_wandb_callback(callbacks_dict, config, state, logger)

        if config["early_stop"]:
            callbacks_dict[FitExperimentBase.EARLY_STOPPING_CALLBACK_NAME] = callbacks.EarlyStopping(score_fn,
                                                                                                     score_name=config["score_metric_name"],
                                                                                                     largest=config["score_largest"],
                                                                                                     min_delta=config["early_stop_min_delta"],
                                                                                                     patience=config["early_stop_patience"],
                                                                                                     cooldown=config["early_stop_cooldown"],
                                                                                                     validate_every=config["validate_every"],
                                                                                                     restore_best_weights=config[
                                                                                                         "early_stop_restore_best_weights"],
                                                                                                     logger=logger)
        self.customize_callbacks(callbacks_dict, model, datamodule, config, state, logger)
        return callbacks.ComposeCallback(callbacks_dict)

    def create_additional_metadata_to_log(self, model: nn.Module, datamodule: DataModule, config: dict, state: dict, logger: logging.Logger) -> dict:
        """
        Creates a dictionary of additional metadata to log at the start of the experiment.
        :param model: PyTorch model.
        :param datamodule: the datamodule.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        :param logger: experiment logger.
        :return: dictionary of metadata to log.
        """
        additional_metadata = {
            "num model parameters": module_utils.get_number_of_parameters(model)
        }

        return additional_metadata

    @staticmethod
    def __create_logging_callback(logger: logging.Logger, config: dict, state: dict, additional_metadata: dict = None):
        if not config["save_logs"] and config["disable_console_log"]:
            return Callback()  # Empty callback for no logging

        return callbacks.ProgressLogger(logger,
                                        train_batch_log_interval=config["train_batch_log_interval"],
                                        epoch_log_interval=config["epoch_log_interval"],
                                        config=config,
                                        additional_metadata=additional_metadata,
                                        context=state["context"])

    def __add_tensorboard_callback(self, callbacks_dict: OrderedDict, config: dict, state: dict):
        # Lazy load tensorboard import to prevent the need for this dependency when not used
        from ..train.callbacks.tensorboard_callback import TensorboardCallback
        callbacks_dict[FitExperimentBase.TENSORBOARD_CALLBACK_NAME] = TensorboardCallback(state["experiment_dir"],
                                                                                          epoch_log_interval=config["epoch_log_interval"])

    def __add_wandb_callback(self, callbacks_dict: OrderedDict, config: dict, state: dict, logger: logging.Logger):
        # Lazy load wandb import to prevent the need for this dependency when not used
        from ..train.callbacks.wandb_callback import WandBCallback
        callbacks_dict[FitExperimentBase.WANDB_CALLBACK_NAME] = WandBCallback(project_name=config["wandb_project_name"],
                                                                              experiment_name=config["experiment_name"],
                                                                              experiment_config=config,
                                                                              entity_name=config["wandb_entity_name"],
                                                                              experiment_start_time=state["experiment_start_time"],
                                                                              track_files_dir=state["experiment_dir"],
                                                                              exclude_files=config["wandb_exclude_files"],
                                                                              epoch_log_interval=config["epoch_log_interval"],
                                                                              track_model=config["wandb_track_model"],
                                                                              resume_id=config["wandb_resume_id"],
                                                                              manual_finish=True,
                                                                              zip_log_files=True,
                                                                              logger=logger)

    def customize_callbacks(self, callbacks_dict: OrderedDict, model: nn.Module, datamodule: DataModule,
                            config: dict, state: dict, logger: logging.Logger):
        """
        Hook for customizing (adding, editing, removing) callbacks.
        :param callbacks_dict: Ordered dictionary of default callbacks.
        :param model: PyTorch model.
        :param datamodule: the datamodule.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        :param logger: experiment logger.
        """
        pass

    @abstractmethod
    def create_trainer(self, model: nn.Module, datamodule: DataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        """
        Creates a Trainer object for training the model.
        :param model: PyTorch model.
        :param datamodule: the datamodule.
        :param train_evaluator: Train evaluator.
        :param val_evaluator: Validation evaluator.
        :param callback: Callback to be called during training.
        :param device: device to use for training.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        :param logger: experiment logger.
        :return: Trainer for training the model.
        """
        raise NotImplementedError

    def create_experiment_result(self, model: nn.Module, datamodule: DataModule, trainer: Trainer, fit_output: FitOutput,
                                 config: dict, state: dict, logger: logging.Logger) -> ExperimentResult:
        """
        Creates the experiment result from the fit output.
        :param model: PyTorch model.
        :param datamodule: the datamodule.
        :param trainer: Trainer object used to train the model.
        :param fit_output: FitOutput that is the result of fitting a Trainer object.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        :param logger: experiment logger.
        :return: ExperimentResult experiment result for the given fit output.
        """
        score_info = self.__get_score_info(config, state)
        if score_info.return_best_score:
            return FitExperimentResultFactory.create_from_best_metric_score(model=model,
                                                                            metric_name=score_info.metric_name,
                                                                            fit_output=fit_output,
                                                                            largest=score_info.largest,
                                                                            is_train_metric=score_info.is_train_metric)

        return FitExperimentResultFactory.create_from_last_metric_score(model=model,
                                                                        metric_name=score_info.metric_name,
                                                                        fit_output=fit_output,
                                                                        largest=score_info.largest,
                                                                        is_train_metric=score_info.is_train_metric)

    def __verify_no_metric_name_collisions(self, train_evaluator: TrainEvaluator, val_evaluator: Evaluator, logger: logging.Logger):
        train_metric_infos = train_evaluator.get_metric_infos()
        val_metric_infos = val_evaluator.get_metric_infos()

        for train_metric_name in train_metric_infos:
            if train_metric_name in val_metric_infos:
                error_msg = f"Metric name collision. Train and Validation metrics of same name '{train_metric_name}' exist, which " \
                            f"can cause unexpected behaviour when reporting metrics."
                logger.error(error_msg)
                raise ValueError(error_msg)

    def on_experiment_end(self, model: nn.Module, datamodule: DataModule, trainer: Trainer, fit_output: FitOutput,
                          experiment_result: ExperimentResult, config: dict, state: dict, logger: logging.Logger):

        """
        Hook at end of experiment, before running cleanup.
        :param model: PyTorch model.
        :param datamodule: the datamodule.
        :param trainer: Trainer object used to train the model.
        :param fit_output: FitOutput that is the result of fitting a Trainer object.
        :param experiment_result: ExperimentResult that will be returned.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        :param logger: experiment logger.
        """
        pass

    def cleanup(self, model: nn.Module, datamodule: DataModule, trainer: Trainer, fit_output: FitOutput, config: dict, state: dict,
                logger: logging.Logger):
        """
        Runs any cleanup code after experiment has finished.
        :param model: PyTorch model.
        :param datamodule: the datamodule.
        :param trainer: Trainer object used to train the model.
        :param fit_output: FitOutput that is the result of fitting a Trainer object.
        :param config: configuration dictionary.
        :param state: dictionary of the experiment state.
        :param logger: experiment logger.
        :return: ExperimentResult experiment result for the given fit output.
        """
        if isinstance(trainer.callback, callbacks.ComposeCallback) and FitExperimentBase.WANDB_CALLBACK_NAME in trainer.callback.callbacks:
            wandb_callback = trainer.callback.callbacks[FitExperimentBase.WANDB_CALLBACK_NAME]
            if wandb_callback.manual_finish:
                wandb_callback.finish()

    def __save_summary(self, experiment_dir: str, experiment_result: ExperimentResult, summary_file_name: str = "summary.json"):
        summary_file_path = os.path.join(experiment_dir, summary_file_name)
        with open(summary_file_path, "w") as f:
            json.dump(experiment_result.summary, f, indent=2)

    def run(self, config: dict, context: dict = None):
        self.validate_config(config)

        state = {"context": context}
        self.initialize(config, state)
        logger = state["logger"]

        try:
            datamodule = self.create_datamodule(config, state, logger)

            model = self.create_model(datamodule, config, state, logger)
            gpu_ids = state["gpu_ids"]
            device = state["device"]

            if device is not None:
                model = model.to(device)

            if len(gpu_ids) > 1:
                model = DataParallelPassthrough(model, device_ids=gpu_ids)

            train_dataloader = self.create_train_dataloader(datamodule, config, state, logger)
            train_evaluator, val_evaluator = self.create_train_and_validation_evaluators(model, datamodule, device, config, state, logger)
            self.__verify_no_metric_name_collisions(train_evaluator, val_evaluator, logger)

            callback = self.create_trainer_callback(model, datamodule, config, state, logger)
            trainer = self.create_trainer(model, datamodule, train_evaluator, val_evaluator, callback, device, config, state, logger)

            if "trainer_checkpoint" in config and config["trainer_checkpoint"]:
                trainer.load_state_dict(torch.load(config["trainer_checkpoint"], map_location=device))

            validate_every = config["validate_every"] if "validate_every" in config else 1
            fit_output = trainer.fit(train_dataloader, num_epochs=config["epochs"], validate_every=validate_every)
            experiment_result = self.create_experiment_result(model, datamodule, trainer, fit_output, config, state, logger)

            self.on_experiment_end(model, datamodule, trainer, fit_output, experiment_result, config, state, logger)

            if config["save_logs"]:
                self.__save_summary(experiment_dir=state["experiment_dir"], experiment_result=experiment_result)

            self.cleanup(model, datamodule, trainer, fit_output, config, state, logger)
            return experiment_result
        except Exception:
            logger.exception("Exception while running experiment.")
            raise
