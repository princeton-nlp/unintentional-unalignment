from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from common.evaluation.evaluators.evaluator import VoidEvaluator, TrainEvaluator, Evaluator
from common.serialization.torch_serializable import TorchSerializable
from common.train.callbacks import ComposeCallback, Callback
from common.train.fit_output import FitOutput
from common.train.stop_fit_iteration import StopFitIteration
from common.train.value_store import ValueStore


class AccelerateTrainer(TorchSerializable, ABC):
    """
    Model train functionality wrapper.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, train_evaluator: TrainEvaluator = VoidEvaluator(),
                 val_evaluator: Evaluator = VoidEvaluator(), callback: Callback = None):
        self.model = model
        self.optimizer = optimizer
        self.train_evaluator = train_evaluator
        self.val_evaluator = val_evaluator
        self.value_store = ValueStore()
        self.callback = callback if callback is not None else ComposeCallback(OrderedDict())
        self.epoch = -1

    @abstractmethod
    def batch_update(self, batch_num: int, batch, total_num_batches: int) -> dict:
        """
        Runs a single train batch update.
        :param batch_num: current batch number (starts from 0).
        :param batch: batch loaded from the data loader.
        :param total_num_batches: number of batches in an epoch.
        :return: output that will be given to the train evaluator. Usually the output will be a dictionary with the loss value and model outputs and
        labels if exists for the batch.
        """
        raise NotImplementedError

    def fit(self, dataloader: torch.utils.data.DataLoader, num_epochs: int = 1, validate_every: int = 1) -> FitOutput:
        """
        Trains model using the training data_loader for the specified number of epochs.
        :param dataloader: training data loader.
        :param num_epochs: number of training epochs.
        :param validate_every: run validation phase every this number of epochs.
        :return: FitOutput object with saved tracked values and information on the fitting process.
        """
        original_train_mode = self.model.training
        try:
            self.callback.on_fit_initialization(self)

            output = FitOutput(self.value_store)
            start_epoch = self.epoch
            self.callback.on_fit_start(self, num_epochs)

            self.__try_with_catching_stop_fit_exception(lambda: self.__fit_for_num_epochs(dataloader, num_epochs, validate_every, output), output)

            output.update_train_tracked_values(self.train_evaluator.get_tracked_values())
            output.update_val_tracked_values(self.val_evaluator.get_tracked_values())
            output.last_epoch = self.epoch
            self.callback.on_fit_end(self, self.epoch - start_epoch, output)
            return output
        except Exception as e:
            self.callback.on_exception(self, e)
            raise
        finally:
            self.model.train(original_train_mode)
            self.callback.on_fit_termination(self)

    def __fit_for_num_epochs(self, dataloader: torch.utils.data.DataLoader, num_epochs: int, validate_every: int, output: FitOutput):
        for i in range(num_epochs):
            self.epoch += 1
            self.callback.on_epoch_start(self)

            # If StopFitIteration is thrown on training or validation, still run epoch end callbacks for graceful exiting
            break_fitting = self.__try_with_catching_stop_fit_exception(lambda: self.__train(dataloader), output)

            if not break_fitting:
                if (self.epoch + 1) % validate_every == 0 or i == num_epochs - 1:
                    break_fitting = self.__try_with_catching_stop_fit_exception(lambda: self.__validate(), output)

            self.callback.on_epoch_train_and_validation_end(self)
            self.callback.on_epoch_end(self)

            if break_fitting:
                break

    def __train(self, dataloader: torch.utils.data.DataLoader):
        self.model.train()
        self.callback.on_epoch_train_start(self, len(dataloader))
        self.train_evaluator.epoch_start(self.epoch)

        for batch_num, batch in enumerate(dataloader):
            self.callback.on_train_batch_start(self, batch_num)

            output = self.batch_update(batch_num, batch, total_num_batches=len(dataloader))
            with torch.no_grad():
                metric_values = self.train_evaluator.evaluate_batch(output)

            self.callback.on_train_batch_end(self, batch_num, output, metric_values)

        self.train_evaluator.epoch_end(self.epoch)
        epoch_train_metric_values = {name: tracked_value.current_value
                                     for name, tracked_value in self.train_evaluator.get_tracked_values().items()
                                     if tracked_value.epoch_last_updated == self.epoch}
        self.callback.on_epoch_train_end(self, epoch_train_metric_values)

    def __validate(self):
        self.model.eval()
        self.callback.on_epoch_validation_start(self)
        self.val_evaluator.epoch_start(self.epoch)

        with torch.no_grad():
            self.val_evaluator.evaluate()

        self.val_evaluator.epoch_end(self.epoch)
        epoch_val_metric_values = {name: tracked_value.current_value
                                   for name, tracked_value in self.val_evaluator.get_tracked_values().items()
                                   if tracked_value.epoch_last_updated == self.epoch}
        self.callback.on_epoch_validation_end(self, epoch_val_metric_values)

    def __try_with_catching_stop_fit_exception(self, callable: Callable, output: FitOutput):
        """
        Executes the callable and catches any StopFitIteration exceptions thrown. Returns True if a StopFitIteration was thrown and False otherwise.
        Can be used for gracefully exiting training after such an exception (calling the rest of the callbacks before stopping).
        """
        try:
            callable()
            return False
        except StopFitIteration as e:
            # If StopFitIteration was thrown (usually by a callback) should exit gracefully the fitting process.
            self.callback.on_exception(self, e)
            output.exception = e
            return True

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "value_store": self.value_store.state_dict(),
            "train_evaluator": self.train_evaluator.state_dict(),
            "val_evaluator": self.val_evaluator.state_dict(),
            "callback": self.callback.state_dict(),
            "epoch": self.epoch
        }

    def load_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.value_store.load_state_dict(state_dict["value_store"])
        self.train_evaluator.load_state_dict(state_dict["train_evaluator"])
        self.val_evaluator.load_state_dict(state_dict["val_evaluator"])
        self.callback.load_state_dict(state_dict["callback"])
        self.epoch = state_dict["epoch"]
