from ..serialization.torch_serializable import TorchSerializable


class TrackedValue(TorchSerializable):
    """
    Tracks a certain value during training, allowing aggregation of per batch and per epoch values.
    """

    def __init__(self, name: str, save_epoch_values: bool = True, num_per_epoch_batch_histories_to_save: int = 0, is_scalar: bool = True):
        """
        :param name: Name of the tracked value.
        :param save_epoch_values: Flag whether or not to accumulate value history through epochs (if it is updated).
        :param num_per_epoch_batch_histories_to_save: Number of last epochs to save the per batch history for (if it is updated). -1 for saving all
        epoch batch histories.
        :param is_scalar: Flag whether the tracked value is a scalar.
        """
        self.name = name
        self.save_epoch_values = save_epoch_values
        self.num_per_epoch_batch_histories_to_save = num_per_epoch_batch_histories_to_save
        self.is_scalar = is_scalar

        self.current_value = None
        self.epoch_last_updated = -1

        self.epoch_values = []
        self.epochs_with_values = []

        self.per_epoch_batch_histories = []
        self.epochs_with_batch_history = []

    def epoch_start(self, epoch_num: int):
        """
        Initializes a new epoch batch history.
        :param epoch_num: Number of starting epoch.
        """
        if self.num_per_epoch_batch_histories_to_save == 0:
            return

        self.per_epoch_batch_histories.append([])
        self.epochs_with_batch_history.append(epoch_num)

        if self.num_per_epoch_batch_histories_to_save != -1 and len(self.per_epoch_batch_histories) > self.num_per_epoch_batch_histories_to_save:
            del self.per_epoch_batch_histories[0]
            del self.epochs_with_batch_history[0]

    def add_batch_value(self, value):
        """
        Adds a batch value to the current epoch, if batch history saving is supported.
        :param value: A batch value.
        """
        if self.num_per_epoch_batch_histories_to_save == 0:
            return

        self.per_epoch_batch_histories[-1].append(value)

    def epoch_end(self, value, epoch_num: int):
        """
        Updates the current value and epoch values.
        :param value: The epoch value.
        :param epoch_num: Number of ending epoch.
        """
        self.current_value = value
        self.epoch_last_updated = epoch_num

        if self.save_epoch_values:
            self.epoch_values.append(value)
            self.epochs_with_values.append(epoch_num)

    def reset_all_history(self):
        """
        Resets current value and all epoch and batch wise history.
        """
        self.current_value = None
        self.epoch_values = []
        self.epochs_with_values = []
        self.per_epoch_batch_histories = []
        self.epochs_with_batch_history = []

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "current_value": self.current_value,
            "epoch_last_updated": self.epoch_last_updated,
            "epoch_values": self.epoch_values,
            "epochs_with_values": self.epochs_with_values,
            "per_epoch_batch_histories": self.per_epoch_batch_histories,
            "epochs_with_batch_history": self.epochs_with_batch_history
        }

    def load_state_dict(self, state_dict: dict):
        self.name = state_dict["name"]
        self.current_value = state_dict["current_value"]
        self.epoch_last_updated = state_dict["epoch_last_updated"]
        self.epoch_values = state_dict["epoch_values"]
        self.epochs_with_values = state_dict["epochs_with_values"]
        self.per_epoch_batch_histories = state_dict["per_epoch_batch_histories"]
        self.epochs_with_batch_history = state_dict["epochs_with_batch_history"]
