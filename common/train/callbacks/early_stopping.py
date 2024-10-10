from __future__ import annotations

import copy
import logging
import operator
from typing import Callable, TYPE_CHECKING

import numpy as np

from .callback import Callback
from ..stop_fit_iteration import StopFitIteration

if TYPE_CHECKING:
    from ..trainer import Trainer


class EarlyStopping(Callback):
    """
    Will stop training when a monitored quantity has stopped improving.
    """

    def __init__(self, score_func: Callable[[Trainer], float], score_name: str = "", largest: bool = True, min_delta: float = 0, patience: int = 0,
                 cooldown: int = 0, validate_every: int = 1, restore_best_weights: bool = False, logger: logging.Logger = None):
        """
        :param score_func: callable that takes a trainer as a parameter and returns a score for it.
        :param score_name: name of the score metric (used for StopFitIteration message).
        :param largest: flag whether largest score value is better, false for smallest.
        :param min_delta: minimum change to be considered an improvement in an epoch.
        :param patience: number of checks with no improvement after which training will be stopped.
        :param cooldown: number of epochs at beginning of training to not check for improvement.
        :param validate_every: epoch interval to validate early stopping condition every this number of epochs.
        :param restore_best_weights: flag whether to restore model weights from the epoch with the best score value. If False, the model weights
        obtained at the last step of training are used.
        :param logger: optional logger to log details such as restoration of best weights.
        """
        self.score_func = score_func
        self.score_name = score_name
        self.score_name_str = self.score_name if self.score_name else "score"
        self.largest = largest
        self.patience = patience
        self.cooldown = cooldown
        self.validate_every = validate_every
        self.restore_best_weights = restore_best_weights
        self.best_score_epoch = -1

        self.best_model_state = None
        self.num_not_improved_in_a_row = 0
        self.min_delta = min_delta if self.largest else -min_delta
        self.best_score = -np.inf if self.largest else np.inf
        self.score_is_better_op = operator.gt if self.largest else operator.lt

        self.logger = logger

    def on_fit_start(self, trainer, num_epochs):
        if self.restore_best_weights:
            self.best_model_state = copy.deepcopy(trainer.model.state_dict())

            if self.logger:
                self.logger.info("EarlyStopping Callback: Saved model state at start of training since 'restore_best_weights' is True.")

    def on_fit_end(self, trainer, num_epochs_ran, fit_output):
        if self.restore_best_weights and self.best_model_state is not None:
            trainer.model.load_state_dict(self.best_model_state)

            if self.logger:
                self.logger.info(f"EarlyStopping Callback: Restored model weights from epoch {self.best_score_epoch} "
                                 f"which attained the best score: {self.score_name_str} = {self.best_score}.")

    def on_epoch_end(self, trainer):
        if trainer.epoch < self.cooldown:
            return

        if (trainer.epoch + 1) % self.validate_every == 0:
            self.__early_stopping_check(trainer)

    def __early_stopping_check(self, trainer):
        cur_score = self.score_func(trainer)
        if self.score_is_better_op(cur_score - self.min_delta, self.best_score):
            self.num_not_improved_in_a_row = 0
            self.best_score = cur_score
            self.best_score_epoch = trainer.epoch

            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(trainer.model.state_dict())

                if self.logger:
                    self.logger.info(f"EarlyStopping Callback: Saved model state at epoch {trainer.epoch} "
                                     f"which attains the best score: {self.score_name_str} = {self.best_score}.")
        else:
            self.num_not_improved_in_a_row += 1

        if self.num_not_improved_in_a_row > self.patience:
            self.__early_stop(trainer.epoch)

    def __early_stop(self, epoch):
        raise StopFitIteration(f"Early stopping at end of epoch {epoch} because {self.score_name_str} has not improved in "
                               f"{self.num_not_improved_in_a_row} validations in a row")
