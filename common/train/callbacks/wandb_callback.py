import glob
import logging
import os
import pathlib
import zipfile
from datetime import datetime

import torch.multiprocessing as multiprocessing
import wandb
import wandb.util

from .callback import *


class WandBCallback(Callback):
    """
    WandB experiment tracker callback. Allows to persist metrics and other experiment related information.
    """

    def __init__(self, project_name: str, experiment_name: str, experiment_config: dict, entity_name: str = "",
                 experiment_start_time: datetime = None, track_files_dir: str = "", exclude_files: Sequence[str] = None, epoch_log_interval: int = 1,
                 track_model: str = None, resume_id: str = "", exclude_metrics: Sequence[str] = None, manual_finish: bool = False,
                 zip_log_files: bool = False, logger: logging.Logger = None):
        """
        :param project_name: name of project (used to group multiple experiments)
        :param experiment_name: name of experiment
        :param experiment_config: configuration of experiment run to persist
        :param entity_name: wandb entity (user/team) to report results under. Default will use logged in user
        :param experiment_start_time: experiment start time in utc to append to experiment name (default is to not append timestamp)
        :param track_files_dir: path to directory with files to track. At end of fitting all files in directory will be saved to WandB, unless they
        are specifically excluded.
        :param exclude_files: sequence of file paths or glob to ignore when uploading files from the tracked files directory
        :param epoch_log_interval: log epoch progress every this number of epochs
        :param track_model: enable tracking model information. Supports: 'gradients', 'parameters', 'all' (see wandb docs)
        :param resume_id: id of wandb run to resume
        :param exclude_metrics: sequence of metric names to exclude from wandb logging
        :param manual_finish: do not finish WandB tracking at end of fitting
        :param zip_log_files: zip log files before uploading to wandb, and don't upload original log files (used to save space)
        :param logger: optional logger to log details such as wandb run path.
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.entity_name = entity_name if entity_name else None
        self.experiment_start_time = experiment_start_time
        if self.experiment_start_time:
            self.experiment_name = f"{self.experiment_name}_{self.experiment_start_time.strftime('%Y_%m_%d-%H_%M_%S')}"

        self.experiment_config = experiment_config
        self.epoch_log_interval = epoch_log_interval
        self.track_files_dir = track_files_dir
        self.exclude_files = exclude_files if exclude_files is not None else set()
        self.track_model = track_model
        self.resume_id = resume_id
        self.exclude_metrics = exclude_metrics if exclude_metrics is not None else set()
        self.manual_finish = manual_finish
        self.zip_log_files = zip_log_files
        self.logger = logger

        self.run = None

    def on_fit_initialization(self, trainer):
        # If using multiprocessing then this will log to wandb from separate thread (and not process) as a daemon process cannot create a new process
        settings = wandb.Settings(start_method="thread") if multiprocessing.current_process().daemon else wandb.Settings()
        wandb_id = wandb.util.generate_id() if not self.resume_id else self.resume_id

        self.run = wandb.init(id=wandb_id,
                              project=self.project_name,
                              name=self.experiment_name,
                              entity=self.entity_name,
                              config=self.experiment_config,
                              reinit=True,
                              resume=False if not self.resume_id else "must",
                              settings=settings)

        if self.logger:
            self.logger.info(f"Initialized WandB tracking\n"
                             f"Entity name: {wandb.run.entity}\n"
                             f"Project name: {wandb.run.project}\n"
                             f"Experiment name: {wandb.run.name}\n"
                             f"Run id: {wandb.run.id}\n"
                             f"Run path: {wandb.run.path}\n"
                             f"Url: {wandb.run.url}\n"
                             f"Resumed from existing run: {wandb.run.resumed}")

    def on_fit_start(self, trainer, num_epochs):
        if self.track_model:
            wandb.watch(trainer.model, log=self.track_model)

    def __get_not_excluded_scalar_metric_values(self, evaluator, metric_values):
        metric_infos = evaluator.get_metric_infos()
        scalar_metric_names = {name for name, metric_info in metric_infos.items() if name not in self.exclude_metrics and metric_info.is_scalar}
        metric_values = {name: value for name, value in metric_values.items() if name in scalar_metric_names}
        return metric_values

    def on_epoch_train_end(self, trainer, metric_values):
        if self.epoch_log_interval > 0 and (trainer.epoch + 1) % self.epoch_log_interval == 0:
            metric_values = self.__get_not_excluded_scalar_metric_values(trainer.train_evaluator, metric_values)
            self.__log_epoch_metrics_to_wandb(trainer, metric_values)

    def on_epoch_validation_end(self, trainer, metric_values):
        if self.epoch_log_interval > 0 and (trainer.epoch + 1) % self.epoch_log_interval == 0:
            metric_values = self.__get_not_excluded_scalar_metric_values(trainer.val_evaluator, metric_values)
            self.__log_epoch_metrics_to_wandb(trainer, metric_values)

    def on_epoch_end(self, trainer):
        if self.epoch_log_interval > 0 and (trainer.epoch + 1) % self.epoch_log_interval == 0:
            self.__log_other_epoch_tracked_values_to_wandb(trainer)

    def __log_other_epoch_tracked_values_to_wandb(self, trainer):
        tracked_values = trainer.value_store.tracked_values
        metric_values = {}
        for name, tracked_value in tracked_values.items():
            if name in self.exclude_metrics or not tracked_value.is_scalar or tracked_value.epoch_last_updated != trainer.epoch:
                continue

            metric_values[name] = tracked_value.current_value

        self.__log_epoch_metrics_to_wandb(trainer, metric_values)

    def __log_epoch_metrics_to_wandb(self, trainer, metric_values: dict):
        wandb_metrics = metric_values.copy()
        wandb_metrics["epoch"] = trainer.epoch
        wandb.log(wandb_metrics, step=trainer.epoch)

    def on_fit_termination(self, trainer):
        if not self.manual_finish:
            self.finish()

    def finish(self):
        if self.run is None:
            return

        try:
            files_in_tracked_dir = [file for file in glob.glob(f"{self.track_files_dir}/**", recursive=True) if not os.path.isdir(file)]
            files_to_exclude = []
            for exclude_file_glob in self.exclude_files:
                files_to_exclude.extend([file for file in glob.glob(f"{self.track_files_dir}/{exclude_file_glob}", recursive=True)
                                         if not os.path.isdir(file)])

            files_to_track = set(files_in_tracked_dir) - set(files_to_exclude)
            for file_path in files_to_track:
                if not self.zip_log_files or not file_path.endswith(".log"):
                    wandb.save(file_path)
                else:
                    log_file_name = pathlib.Path(file_path)
                    with zipfile.ZipFile(log_file_name.with_suffix(".zip"), "w") as zip_file:
                        zip_file.write(log_file_name)

                    wandb.save(str(log_file_name.with_suffix(".zip")))

            if self.logger:
                self.logger.info(f"Finished uploading {len(files_to_track)} tracked files in experiment directory to WandB successfully")
        finally:
            if self.run is not None:
                self.run.finish()
                self.run = None
