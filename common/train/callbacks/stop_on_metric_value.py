from __future__ import annotations

import operator
from typing import Callable
from typing import TYPE_CHECKING

from common.train.callbacks import Callback
from common.train.stop_fit_iteration import StopFitIteration

if TYPE_CHECKING:
    from ..trainer import Trainer


class StopOnMetricValue(Callback):
    """
    Stops training if a metric crosses some value.
    """

    def __init__(self, metric_name: str, is_train_metric: bool, threshold_value: float, largest: bool, cooldown: int = 0,
                 patience: int = 10, validate_every: int = 1):
        """
        :param metric_name: name of the metric.
        :param is_train_metric: needs to be True if the metric is a training metric and False for validation metric.
        :param threshold_value: value that if the metric remains above/below training will be stopped. 
        :param largest: if True then training will be stopped if the value remains above the given value, otherwise training will be stopped if 
        the value remains below the given value.
        :param cooldown: number of epochs from start of training before checking whether to stop.
        :param patience: number of epochs metric has to remain above/below the threshold before stopping.
        :param validate_every: epoch interval to validate stopping condition every this number of epochs.
        """
        self.metric_name = metric_name
        self.is_train_metric = is_train_metric
        self.metric_value_fn = self.__create_metric_value_fn(metric_name, is_train_metric)
        self.threshold_value = threshold_value
        self.largest = largest
        self.cooldown = cooldown
        self.patience = patience
        self.validate_every = validate_every

        self.value_passed_threshold_op = operator.ge if self.largest else operator.le
        self.num_beyond_threshold_in_a_row = 0

    def __create_metric_value_fn(self, metric_name: str, is_train_metric: bool) -> Callable[[Trainer], float]:
        def metric_value_fn(trainer: Trainer):
            evaluator = trainer.train_evaluator if is_train_metric else trainer.val_evaluator
            return evaluator.get_tracked_values()[metric_name].current_value

        return metric_value_fn

    def on_epoch_end(self, trainer):
        if trainer.epoch < self.cooldown:
            return

        if (trainer.epoch + 1) % self.validate_every == 0:
            self.__check_metric_value_beyond_threshold(trainer)

    def __check_metric_value_beyond_threshold(self, trainer):
        curr_value = self.metric_value_fn(trainer)
        if self.value_passed_threshold_op(curr_value, self.threshold_value):
            self.num_beyond_threshold_in_a_row += 1
        else:
            self.num_beyond_threshold_in_a_row = 0

        if self.num_beyond_threshold_in_a_row > self.patience:
            self.__stop_fitting(trainer.epoch)

    def __stop_fitting(self, epoch):
        raise StopFitIteration(f"Stopping at end of epoch {epoch} because {self.metric_name} was {'above' if self.largest else 'below'}  "
                               f"{self.threshold_value} for at least {self.num_beyond_threshold_in_a_row} epochs validated in a row")
