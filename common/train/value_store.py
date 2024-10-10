from typing import Dict

from .tracked_value import TrackedValue
from ..serialization.torch_serializable import TorchSerializable


class ValueStore(TorchSerializable):
    """
    Value store that allows storing of TrackedValues as well as any other auxiliary values.
    """

    def __init__(self):
        self.tracked_values = {}
        self.other_values = {}

    def get_tracked_values_with_history(self) -> Dict[str, TrackedValue]:
        """
        :return: Dict of TrackedValue objects that save epoch history.
        """
        return {name: tracked_value for name, tracked_value in self.tracked_values.items() if tracked_value.save_epoch_values}

    def get_tracked_value(self, name: str) -> TrackedValue:
        """
        Gets a tracked value if it exists, None otherwise.
        :param name: Name of the tracked value.
        """
        return self.tracked_values.get(name)

    def tracked_value_exists(self, name: str) -> bool:
        """
        Checks if there exists a tracked value with the given name.
        :param name: Name of the tracked value.
        :return: True if a tracked value of the given name exists, False otherwise.
        """
        return name in self.tracked_values

    def add_tracked_value(self, tracked_value: TrackedValue):
        """
        Adds a tracked value to the tracked values. Raises a ValueError if a TrackedValue with the given name already exists.
        :param tracked_value: TrackedValue object.
        """
        if tracked_value.name in self.tracked_values:
            raise ValueError(f"Failed to add a tracked value to ValueStore. Tracked value with name '{tracked_value.name}' already exists.")

        self.tracked_values[tracked_value.name] = tracked_value

    def get_other_value(self, name: str):
        """
        Gets a value from the other values it exists, None otherwise.
        :param name: Name of the value.
        """
        return self.other_values.get(name)

    def other_value_exists(self, name: str) -> bool:
        """
        Checks if there exists an other value with the given name.
        :param name: Name of the other value.
        :return: True if an other value of the given name exists, False otherwise.
        """
        return name in self.other_values

    def add_other_value(self, name: str, value):
        """
        Adds a value to the other values store. Raises a ValueError if a value with the given name already exists.
        :param name: Name of the value.
        :param value: A value to store.
        """
        if name in self.other_values:
            raise ValueError(f"Failed to add a value to other values in a ValueStore. Value with name '{name}' already exists.")

        self.other_values[name] = value

    def update_other_value(self, name: str, value):
        """
        Updates an existing value in the other values store.
        :param name: Name of the value to update.
        :param value: The new value.
        """
        if name not in self.other_values:
            raise ValueError(f"Failed to update a value in other values of a ValueStore. Value with name '{name}' does not exist.")

        self.other_values[name] = value

    def state_dict(self) -> dict:
        tracked_values_state = {name: tracked_value.state_dict() for name, tracked_value in self.tracked_values.items()}
        other_values_state = {}
        for name, value in other_values_state.items():
            value_state = value.state_dict() if isinstance(value, TorchSerializable) else value
            other_values_state[name] = value_state

        return {
            "tracked_values": tracked_values_state,
            "other_values": other_values_state
        }

    def load_state_dict(self, state_dict: dict):
        self.__class__.__load_tracked_values(self.tracked_values, state_dict["tracked_values"])
        self.__class__.__load_other_values(self.other_values, state_dict["other_values"])

    @staticmethod
    def __load_tracked_values(tracked_values_dict, tracked_values_states_dict):
        for name, tracked_value_state in tracked_values_states_dict.items():
            if name not in tracked_values_dict:
                tracked_values_dict[name] = TrackedValue(name)
            tracked_values_dict[name].load_state_dict(tracked_value_state)

    @staticmethod
    def __load_other_values(other_values_dict, other_values_state_dict):
        for name, other_value_state in other_values_state_dict.items():
            if name in other_values_dict and isinstance(other_values_dict[name], TorchSerializable):
                other_values_dict[name].load_state_dict(other_value_state)
            else:
                other_values_dict[name] = other_value_state
