from __future__ import annotations

from typing import Callable

import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from .callback import *

if TYPE_CHECKING:
    from ..trainer import Trainer


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when a score has stopped improving.
    """

    def __init__(self, optimizer: optim.Optimizer, score_func: Callable[[Trainer], float], largest: bool = True, factor: float = 0.1,
                 patience: int = 10, threshold: float = 1e-4, threshold_mode: str = "abs", min_lr: float = 0, validate_every: int = 1,
                 logger=None):
        """
        :param optimizer: model optimizer to reduce lr of.
        :param score_func: callable that takes a trainer as a parameter and returns a score for it.
        :param largest: flag whether largest score value is better, false for smallest.
        :param factor: factor to reduce lr by (lr is multiplied by factor).
        :param patience: number of epochs with no improvement after which lr will be reduced.
        :param threshold: minimum change to be considered an improvement in an epoch.
        :param threshold_mode: "abs" for additive improvement, "rel" for multiplicative improvement.
        :param min_lr: minimum possible lr.
        :param validate_every: epoch interval to call step for the ReduceLROnPlateau scheduler.
        :param logger: Logger to use for logging learning rates.
        """
        self.optimizer = optimizer
        self.score_func = score_func
        self.lr_scheduler = scheduler.ReduceLROnPlateau(optimizer, mode="max" if largest else "min", factor=factor, patience=patience,
                                                        threshold=threshold, threshold_mode=threshold_mode, min_lr=min_lr)
        self.validate_every = validate_every
        self.logger = logger

    def on_epoch_end(self, trainer):
        if (trainer.epoch + 1) % self.validate_every == 0:
            cur_score = self.score_func(trainer)
            self.lr_scheduler.step(cur_score)

            if self.logger is not None:
                learning_rates = [param_group["lr"] for param_group in self.lr_scheduler.optimizer.param_groups]
                self.logger.info(f"Learning rate scheduler step done at the end of epoch {trainer.epoch}]. "
                                 f"Current learning rates are: {learning_rates}")

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
