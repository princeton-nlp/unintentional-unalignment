from __future__ import annotations

import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from .callback import *

if TYPE_CHECKING:
    from ..trainer import Trainer


class TensorboardCallback(Callback):
    """
    Reports Tensorboard summaries for scalar metrics.
    """

    def __init__(self, output_dir, create_dir=True, exclude_metrics: Sequence[str] = None, epoch_log_interval: int = 1):
        """
        :param output_dir: output dir of tensorboard logs.
        :param create_dir: create output directory if is not exist.
        :param exclude_metrics: sequence of metric names to exclude from tensorboard.
        :param epoch_log_interval: log epoch progress every this number of epochs
        """
        self.output_dir = output_dir
        self.tensorboard_dir = os.path.join(self.output_dir, "tensorboard")
        self.metric_writers = {}

        self.create_dir = create_dir
        self.exclude_metrics = exclude_metrics if exclude_metrics is not None else set()
        self.epoch_log_interval = epoch_log_interval

    @staticmethod
    def __escape_metric_name(metric_name):
        return metric_name.lower().replace(" ", "_")

    def on_fit_initialization(self, trainer):
        if self.create_dir and not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)

    def __get_not_excluded_scalar_metric_infos(self, evaluator):
        metric_infos = evaluator.get_metric_infos()
        metric_infos = {name: metric_info for name, metric_info in metric_infos.items() if name not in self.exclude_metrics and metric_info.is_scalar}
        return metric_infos

    def on_epoch_train_end(self, trainer, metric_values):
        if self.epoch_log_interval > 0 and (trainer.epoch + 1) % self.epoch_log_interval == 0:
            self.__write_metrics(trainer.train_evaluator, metric_values, trainer.epoch)

    def on_epoch_validation_end(self, trainer, metric_values):
        if self.epoch_log_interval > 0 and (trainer.epoch + 1) % self.epoch_log_interval == 0:
            self.__write_metrics(trainer.val_evaluator, metric_values, trainer.epoch)

    def __write_metrics(self, evaluator, metric_values, epoch):
        metric_infos = self.__get_not_excluded_scalar_metric_infos(evaluator)
        metric_values = {metric_name: value for metric_name, value in metric_values.items() if metric_name in metric_infos}

        for metric_name, metric_value in metric_values.items():
            metric_info = metric_infos[metric_name]
            metric_writer = self.__get_or_register_metric_writer(metric_name)
            metric_writer.add_scalar(metric_info.tag, metric_value, global_step=epoch)

    def on_epoch_end(self, trainer: Trainer):
        if self.epoch_log_interval > 0 and (trainer.epoch + 1) % self.epoch_log_interval == 0:
            self.__write_other_tracked_values(trainer.value_store.tracked_values, trainer.epoch)

    def __write_other_tracked_values(self, tracked_values, epoch):
        tracked_values = {name: tracked_value for name, tracked_value in tracked_values.items() if
                          name not in self.exclude_metrics and tracked_value.is_scalar}

        for name, tracked_value in tracked_values.items():
            if tracked_value.epoch_last_updated != epoch:
                continue

            metric_writer = self.__get_or_register_metric_writer(name)
            metric_writer.add_scalar(name, tracked_value.current_value, global_step=epoch)

    def __get_or_register_metric_writer(self, metric_name):
        escaped_metric_name = self.__escape_metric_name(metric_name)
        if escaped_metric_name not in self.metric_writers:
            summary_path = Path(os.path.join(self.tensorboard_dir, escaped_metric_name)).as_posix()
            self.metric_writers[escaped_metric_name] = SummaryWriter(summary_path)

        return self.metric_writers[escaped_metric_name]
