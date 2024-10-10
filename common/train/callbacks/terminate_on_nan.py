import numbers

import numpy as np
import torch

from .callback import *
from .. import consts as consts
from ..stop_fit_iteration import StopFitIteration


class TerminateOnNaN(Callback):
    """
    Callback that terminates training when a NaN output is encountered in the batch output or in the metric values.
    """

    def __init__(self, verify_batches=True, batch_output_transform=lambda x: []):
        """
        :param verify_batches: Whether to verify also batch outputs or not.
        :param batch_output_transform: transforms the batch output from the trainer batch_update method into a sequence of tensors/numbers.
        """
        self.batch_output_transform = batch_output_transform
        self.verify_batches = verify_batches

    def on_train_batch_end(self, trainer, batch_num, batch_output, metric_values):
        if self.verify_batches:
            outputs = self.batch_output_transform(batch_output)
            self.__verify_outputs(trainer, outputs)
            self.__verify_metric_values(trainer, metric_values, consts.TRAIN_PHASE)

    def on_epoch_train_end(self, trainer, metric_values):
        self.__verify_metric_values(trainer, metric_values, consts.TRAIN_PHASE)

    def on_epoch_validation_end(self, trainer, metric_values):
        self.__verify_metric_values(trainer, metric_values, consts.VALIDATION_PHASE)

    @staticmethod
    def __verify_outputs(trainer, outputs):
        for output in outputs:
            if isinstance(output, numbers.Number):
                output = torch.tensor(output)

            if isinstance(output, torch.Tensor) and not bool(torch.isfinite(output).all()):
                raise StopFitIteration(f"NaN value found in batch outputs. Exiting fitting on epoch {trainer.epoch}")

    @staticmethod
    def __verify_metric_values(trainer, metric_values, phase):
        for name, value in metric_values.items():
            if np.isnan(value) or np.isinf(value):
                raise StopFitIteration(f"{phase} metric '{name}' with NaN value {value} encountered. Exiting fitting on epoch {trainer.epoch}")
