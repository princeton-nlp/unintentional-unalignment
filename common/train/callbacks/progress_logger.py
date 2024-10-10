import json
import logging
import os
import sys
import time
from datetime import datetime

from .callback import *
from ...utils import logging as logging_utils


class ProgressLogger(Callback):
    """
    Logs progress of fitting to the given logger.
    """

    def __init__(self, logger: logging.Logger, train_batch_log_interval: int = 1, epoch_log_interval: int = 1, config: dict = None,
                 additional_metadata: dict = None, context: dict = None):
        self.logger = logger
        self.train_batch_log_interval = train_batch_log_interval
        self.epoch_log_interval = epoch_log_interval
        self.config = config
        self.additional_metadata = additional_metadata
        self.context = context

        self.fit_start_epoch = None
        self.num_epochs = None
        self.num_batches_in_epoch = None
        self.fit_start_time = None
        self.epoch_start_time = None
        self.train_batch_start_time = None
        self.epoch_validation_start_time = None

    def on_fit_start(self, trainer, num_epochs):
        self.fit_start_epoch = trainer.epoch
        self.num_epochs = num_epochs
        self.fit_start_time = datetime.utcnow()
        if self.config is not None:
            self.logger.info(f"Config:\n{json.dumps(self.config, indent=2)}")

        if self.context is not None:
            self.logger.info(f"Context:\n{json.dumps(self.context, indent=2)}")

        if self.additional_metadata is not None:
            self.logger.info(f"Additional Metadata:\n{json.dumps(self.additional_metadata, indent=2)}")

        self.logger.info(f"Starting fit for {num_epochs} epochs. "
                         f"Epochs range (inclusive): {self.fit_start_epoch + 1}--{self.fit_start_epoch + num_epochs}")

    def on_fit_end(self, trainer, num_epochs_ran, fit_output):
        fit_end_time = datetime.utcnow()
        fit_time_delta = fit_end_time - self.fit_start_time
        self.logger.info(f"Finished fit for {num_epochs_ran} epochs. Time took: {fit_time_delta}")

    def on_epoch_start(self, trainer):
        self.epoch_start_time = datetime.utcnow()

    def on_epoch_end(self, trainer):
        if self.epoch_log_interval > 0 and (trainer.epoch + 1) % self.epoch_log_interval == 0:
            epoch_end_time = datetime.utcnow()
            epoch_time_delta = epoch_end_time - self.epoch_start_time
            updated_other_tracked_values = {name: tracked_value.current_value for name, tracked_value in trainer.value_store.tracked_values.items()
                                            if tracked_value.epoch_last_updated == trainer.epoch}

            num_epochs_passed = trainer.epoch - self.fit_start_epoch

            log_msg = f"Finished epoch {trainer.epoch} (fit epoch progress: {num_epochs_passed}/{self.num_epochs}). Time took: {epoch_time_delta}"
            if len(updated_other_tracked_values) > 0:
                log_msg += f"\nAdditional tracked values:\n{json.dumps(updated_other_tracked_values, indent=2)}"

            self.logger.info(log_msg)

    def on_epoch_train_start(self, trainer, num_batches):
        self.num_batches_in_epoch = num_batches

    def on_epoch_train_end(self, trainer, metric_values):
        if self.epoch_log_interval > 0 and (trainer.epoch + 1) % self.epoch_log_interval == 0:
            epoch_train_end_time = datetime.utcnow()
            epoch_train_time_delta = epoch_train_end_time - self.epoch_start_time
            self.logger.info(f"Epoch {trainer.epoch} - Finished train step. Time took: {epoch_train_time_delta}\n"
                             f"Metric values:\n{json.dumps(metric_values, indent=2)}")

    def on_train_batch_start(self, trainer, batch_num):
        self.train_batch_start_time = datetime.utcnow()

    def on_train_batch_end(self, trainer, batch_num, batch_output, metric_values):
        if self.train_batch_log_interval > 0 and (batch_num + 1) % self.train_batch_log_interval == 0:
            train_batch_end_time = datetime.utcnow()
            train_batch_time_delta = train_batch_end_time - self.train_batch_start_time
            self.logger.info(
                f"Epoch {trainer.epoch} - Finished train batch {batch_num + 1}/{self.num_batches_in_epoch}. Time took: {train_batch_time_delta}\n"
                f"Metric values:\n{json.dumps(metric_values, indent=2)}")

    def on_epoch_validation_start(self, trainer):
        self.epoch_validation_start_time = datetime.utcnow()

    def on_epoch_validation_end(self, trainer, metric_values):
        if self.epoch_log_interval > 0 and (trainer.epoch + 1) % self.epoch_log_interval == 0:
            epoch_validation_end_time = datetime.utcnow()
            epoch_validation_time_delta = epoch_validation_end_time - self.epoch_validation_start_time
            self.logger.info(f"Epoch {trainer.epoch} - Finished validation step. Time took: {epoch_validation_time_delta}\n"
                             f"Metric values:\n{json.dumps(metric_values, indent=2)}")

    def on_exception(self, trainer, exception):
        self.logger.exception("Exception while executing fit function")


class FileProgressLogger(ProgressLogger):
    """
    Logs progress of fitting to a file.
    """

    def __init__(self, output_dir: str, experiment_name: str, create_dir: bool = True, add_time_stamp_to_log_name: bool = True,
                 experiment_start_time: datetime = None, msg_format: str = "%(asctime)s - %(levelname)s - %(message)s", log_level: int = logging.INFO,
                 train_batch_log_interval: int = 1, epoch_log_interval: int = 1, config: dict = None, additional_metadata: dict = None,
                 context: dict = None, save_config_separately: bool = True, config_file_name: str = "config.json",
                 context_file_name: str = "context.json", disable_console_log: bool = False):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.create_dir = create_dir
        self.save_config_separately = save_config_separately
        self.config_file_name = config_file_name
        self.config_file_path = os.path.join(output_dir, self.config_file_name) if self.save_config_separately and config is not None else ""
        self.context_file_name = context_file_name
        self.context_file_path = os.path.join(output_dir, self.context_file_name) if self.save_config_separately and context is not None else ""

        self.msg_format = msg_format
        self.log_level = log_level
        self.disable_console_log = disable_console_log

        self.experiment_start_time = experiment_start_time if experiment_start_time is not None else datetime.utcnow()

        if self.config_file_path:
            with open(self.config_file_path, "w") as f:
                json.dump(config, f, indent=2)

        if self.context_file_path:
            with open(self.context_file_path, "w") as f:
                json.dump(context, f, indent=2)

        logger = logging_utils.create_logger(console_logging=not self.disable_console_log,
                                             file_logging=True,
                                             log_dir=self.output_dir,
                                             log_file_name_prefix=self.experiment_name,
                                             create_dir=self.create_dir,
                                             add_time_stamp_to_log_name=add_time_stamp_to_log_name,
                                             timestamp=self.experiment_start_time,
                                             msg_format=self.msg_format,
                                             log_level=self.log_level)

        super().__init__(logger, train_batch_log_interval, epoch_log_interval, config, additional_metadata, context)


class ConsoleProgressLogger(ProgressLogger):
    """
    Logs progress of fitting to the console.
    """

    def __init__(self, msg_format: str = "%(asctime)s - %(levelname)s - %(message)s", log_level: int = logging.INFO,
                 train_batch_log_interval: int = 1, epoch_log_interval: int = 1, config: dict = None,
                 additional_metadata: dict = None, context: dict = None):
        self.msg_format = msg_format
        self.log_level = log_level

        now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
        self.logger_name = f"{now_utc_str}"
        logger = self.__create_console_logger()

        super().__init__(logger, train_batch_log_interval, epoch_log_interval, config, additional_metadata, context)

    def __create_console_logger(self):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.log_level)

        ch = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(self.msg_format)
        formatter.converter = time.gmtime
        ch.setFormatter(formatter)

        logger.addHandler(ch)
        return logger
