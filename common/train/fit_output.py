from typing import Dict

from .tracked_value import TrackedValue
from .value_store import ValueStore


class FitOutput:
    """
    Output object of a Trainer fit method. Contains tracked values for metrics, an additional value store and information on an exception if one
    occurred during fitting.
    """

    def __init__(self, value_store: ValueStore, train_tracked_values: Dict[str, TrackedValue] = None,
                 val_tracked_values: Dict[str, TrackedValue] = None, exception: Exception = None):
        self.train_tracked_values = train_tracked_values if train_tracked_values is not None else {}
        self.val_tracked_values = val_tracked_values if val_tracked_values is not None else {}
        self.value_store = value_store
        self.last_epoch = -1
        self.exception = exception

    def update_train_tracked_values(self, tracked_values: Dict[str, TrackedValue]):
        """
        Updates the train tracked values with the given tracked values. Will raise a ValueError if a train tracked value with the given name
        already exists.
        :param tracked_values: Dictionary of name to TrackedValue.
        """
        FitOutput.__update_tracked_values(self.train_tracked_values, tracked_values, "train")

    def update_val_tracked_values(self, tracked_values: Dict[str, TrackedValue]):
        """
        Updates the validation tracked values with the given tracked values. Will raise a ValueError if a train tracked value with the given name
        already exists.
        :param tracked_values: Dictionary of name to TrackedValue.
        """
        FitOutput.__update_tracked_values(self.val_tracked_values, tracked_values, "validation")

    def exception_occured(self):
        return self.exception is not None

    @staticmethod
    def __update_tracked_values(tracked_values: Dict[str, TrackedValue], new_tracked_values: Dict[str, TrackedValue], phase: str):
        for name in new_tracked_values.keys():
            if name in tracked_values:
                raise ValueError(f"Failed to update the {phase} tracked values. TrackedValue with name '{name}' already exists.")

        tracked_values.update(new_tracked_values)
