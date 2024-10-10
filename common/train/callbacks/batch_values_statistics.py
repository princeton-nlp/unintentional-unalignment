from __future__ import annotations

from typing import Dict

import numpy as np

from .callback import *
from ..tracked_value import TrackedValue

if TYPE_CHECKING:
    from ..trainer import Trainer


class BatchValuesStatistics(Callback):
    """
    Adds per batch statistics for TrackedValues (that save batch history) of the train and validation evaluators, as well as existing other
    TrackedValues in the trainer Value Store. The statistic is added as a TrackedValue to the TrackedValues in the trainer ValueStore. For example,
    this allows tracking of the mean/median/max/min/std of the per batch values for existing metrics.
    """

    def __init__(self, stat="mean", create_only_for=None, exclude=None):
        """
        :param stat: Name code of the statistic to create. Currently supports: 'mean', 'median', 'max', 'min', 'std'.
        :param create_only_for: List of TrackedValue names. If specified then the statistics will be created only for the specified names. Otherwise,
        it will be created for all existing TrackedValues.
        :param exclude: Sequence of TrackedValue names to ignore and not create the statistic for.
        """
        self.stat = stat
        self.stat_func = BatchValuesStatistics.__get_stat_func(stat)
        self.create_only_for = create_only_for if create_only_for is not None else []
        self.exclude = exclude if exclude is not None else []
        self.track_stat_for_tracked_values = {}
        self.stat_tracked_values = {}

    @staticmethod
    def __get_stat_func(stat: str):
        if stat == "mean":
            return np.mean
        elif stat == "median":
            return np.median
        elif stat == "max":
            return np.max
        elif stat == "min":
            return np.min
        elif stat == "std":
            return np.std

        raise ValueError(f"Unsupported score reduction type: {stat}. Supported types are: 'mean', 'median', 'max', 'min', 'std'.")

    @staticmethod
    def __create_statistic_tracked_value_name(name: str, stat: str):
        return f"batch {name} {stat}"

    def on_fit_start(self, trainer: Trainer, num_epochs: int):
        self.__register_batch_statistics_tracked_values_for(trainer, trainer.value_store.tracked_values)
        self.__register_batch_statistics_tracked_values_for(trainer, trainer.train_evaluator.get_tracked_values())
        self.__register_batch_statistics_tracked_values_for(trainer, trainer.val_evaluator.get_tracked_values())

    def __register_batch_statistics_tracked_values_for(self, trainer: Trainer, tracked_values: Dict[str, TrackedValue]):
        for name, tracked_value in tracked_values.items():
            if not self.__track_stat_for(tracked_value):
                continue

            stat_tracked_value_name = BatchValuesStatistics.__create_statistic_tracked_value_name(name, self.stat)
            self.stat_tracked_values[stat_tracked_value_name] = tracked_value

            existing_stat_tracked_value = trainer.value_store.get_tracked_value(stat_tracked_value_name)
            if existing_stat_tracked_value is not None:
                self.track_stat_for_tracked_values[stat_tracked_value_name] = existing_stat_tracked_value
                continue

            stat_tracked_value = TrackedValue(stat_tracked_value_name)
            trainer.value_store.add_tracked_value(stat_tracked_value)
            self.track_stat_for_tracked_values[stat_tracked_value_name] = stat_tracked_value

    def __track_stat_for(self, tracked_value: TrackedValue):
        if tracked_value.num_per_epoch_batch_histories_to_save == 0 or not tracked_value.is_scalar:
            return False

        if tracked_value.name in self.exclude:
            return False

        if self.create_only_for and tracked_value.name not in self.create_only_for:
            return False

        return True

    def on_epoch_start(self, trainer: Trainer):
        for stat_tracked_value in self.track_stat_for_tracked_values.values():
            stat_tracked_value.epoch_start(trainer.epoch)

    def on_epoch_train_and_validation_end(self, trainer: Trainer):
        for stat_tracked_value_name, stat_tracked_value in self.track_stat_for_tracked_values.items():
            original_tracked_value = self.stat_tracked_values[stat_tracked_value_name]

            if original_tracked_value.epochs_with_batch_history[-1] == trainer.epoch:
                stat_value = self.stat_func(original_tracked_value.per_epoch_batch_histories[-1])
                stat_tracked_value.epoch_end(stat_value, trainer.epoch)
