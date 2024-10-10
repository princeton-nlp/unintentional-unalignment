from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Sequence, Union

from ..fit_output import FitOutput
from ...serialization.torch_serializable import TorchSerializable

if TYPE_CHECKING:
    from ..trainer import Trainer


class Callback(TorchSerializable):
    """
    Callback for trainer to allow hooks for added functionality. Callback can raise StopFitIteration during hooks (except on_fit_end and
    on_exception) in order to stop the fitting process.
    """

    def on_fit_initialization(self, trainer: Trainer):
        """
        Called at start of fit process (before on_fit_start). E.g. can be used to initialize connections.
        :param trainer: the executing trainer.
        """
        pass

    def on_fit_start(self, trainer: Trainer, num_epochs: int):
        """
        Called on the start of the fit function.
        :param trainer: the executing trainer.
        :param num_epochs: number of epochs the fit will run for (may be stopped earlier, e.g. due to early stopping or error).
        """
        pass

    def on_epoch_start(self, trainer: Trainer):
        """
        Called on start of each epoch.
        :param trainer: the executing trainer.
        """
        pass

    def on_epoch_train_start(self, trainer: Trainer, num_batches: int):
        """
        Called on the start of the training phase of each epoch.
        :param trainer: the executing trainer.
        :param num_batches: number of batches in the training epoch.
        """
        pass

    def on_train_batch_start(self, trainer: Trainer, batch_num: int):
        """
        Called on train batch start.
        :param trainer: the executing trainer.
        :param batch_num: current batch number.
        """
        pass

    def on_train_batch_end(self, trainer: Trainer, batch_num: int, batch_output, metric_values):
        """
        Called on train batch end.
        :param trainer: the executing trainer.
        :param batch_num: current batch number.
        :param batch_output: output from the update batch trainer function.
        :param metric_values: metric values from the train evaluator.
        """
        pass

    def on_epoch_train_end(self, trainer: Trainer, metric_values):
        """
        Called at the end of the training phase of each epoch.
        :param trainer: the executing trainer.
        :param metric_values: metric values for the training epoch.
        """
        pass

    def on_epoch_validation_start(self, trainer: Trainer):
        """
        Called on validation start.
        :param trainer: the executing trainer.
        """
        pass

    def on_epoch_validation_end(self, trainer: Trainer, metric_values):
        """
        Called on validation end.
        :param trainer: the executing trainer.
        :param metric_values: validation evaluator metric values.
        """
        pass

    def on_epoch_train_and_validation_end(self, trainer: Trainer):
        """
        Called after both training and validation phases have run (even when no validation is done in the epoch). Runs before on_epoch_end, and
        allows e.g. to compute values that depend on both training and validation, but need to run before other callbacks that use on_epoch_end.
        :param trainer: the executing trainer.
        """
        pass

    def on_epoch_end(self, trainer: Trainer):
        """
        Called on the end of each epoch, after the epoch counter has increased.
        :param trainer: the executing trainer.
        """
        pass

    def on_fit_end(self, trainer: Trainer, num_epochs_ran: int, fit_output: FitOutput):
        """
        Called at the end of the fit function.
        :param trainer: the executing trainer.
        :param num_epochs_ran: the number of training epochs ran.
        :param fit_output: output from the fit function containing the training and validation evaluation metrics. Can be updated/changed in this
        callback.
        """
        pass

    def on_exception(self, trainer: Trainer, exception: Exception):
        """
        Called in case of an exception during the fit function.
        :param trainer: the executing trainer.
        :param exception: the exception raised.
        """
        pass

    def on_fit_termination(self, trainer: Trainer):
        """
        Called right before the fit is terminated, both in case of successful completion and in case of exception. E.g. can be used for
        closing open connections.
        :param trainer: the executing trainer.
        """
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict):
        pass


class ComposeCallback(Callback):
    """
    Composes callbacks sequentially. Used to call multiple callbacks sequentially.
    """

    def __init__(self, callbacks: Union[OrderedDict[Union[int, str], Callback], Sequence[Callback]]):
        """
        :param callbacks: Ordered dict of callbacks or sequence of callbacks.
        """
        if not isinstance(callbacks, OrderedDict):
            self.callbacks = OrderedDict()
            for i, callback in enumerate(callbacks):
                self.callbacks[i] = callback
        else:
            self.callbacks = callbacks

    def on_fit_initialization(self, trainer: Trainer):
        for callback in self.callbacks.values():
            callback.on_fit_initialization(trainer)

    def on_fit_start(self, trainer, num_epochs):
        for callback in self.callbacks.values():
            callback.on_fit_start(trainer, num_epochs)

    def on_epoch_start(self, trainer):
        for callback in self.callbacks.values():
            callback.on_epoch_start(trainer)

    def on_epoch_train_start(self, trainer, num_batches):
        for callback in self.callbacks.values():
            callback.on_epoch_train_start(trainer, num_batches)

    def on_train_batch_start(self, trainer, batch_num):
        for callback in self.callbacks.values():
            callback.on_train_batch_start(trainer, batch_num)

    def on_train_batch_end(self, trainer, batch_num, batch_output, metric_values):
        for callback in self.callbacks.values():
            callback.on_train_batch_end(trainer, batch_num, batch_output, metric_values)

    def on_epoch_train_end(self, trainer, metric_values):
        for callback in self.callbacks.values():
            callback.on_epoch_train_end(trainer, metric_values)

    def on_epoch_validation_start(self, trainer):
        for callback in self.callbacks.values():
            callback.on_epoch_validation_start(trainer)

    def on_epoch_validation_end(self, trainer, metric_values):
        for callback in self.callbacks.values():
            callback.on_epoch_validation_end(trainer, metric_values)

    def on_epoch_train_and_validation_end(self, trainer):
        for callback in self.callbacks.values():
            callback.on_epoch_train_and_validation_end(trainer)

    def on_epoch_end(self, trainer):
        for callback in self.callbacks.values():
            callback.on_epoch_end(trainer)

    def on_fit_end(self, trainer, num_epochs_ran, fit_output):
        for callback in self.callbacks.values():
            callback.on_fit_end(trainer, num_epochs_ran, fit_output)

    def on_exception(self, trainer, exception):
        for callback in self.callbacks.values():
            callback.on_exception(trainer, exception)

    def on_fit_termination(self, trainer: Trainer):
        for callback in self.callbacks.values():
            callback.on_fit_termination(trainer)

    def state_dict(self):
        return {name: callback.state_dict() for name, callback in self.callbacks.items()}

    def load_state_dict(self, state_dict):
        for name, callback in self.callbacks.items():
            if name in state_dict:
                callback.load_state_dict(state_dict[name])
