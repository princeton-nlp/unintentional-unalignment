from abc import ABC, abstractmethod

import torch.utils.data


class DataModule(ABC):
    """
    Encapsulates handling of a dataset, including loading, preparation, and dataloader creation.
    """

    @abstractmethod
    def setup(self):
        """
        Runs any setup code necessary for loading and preparing the data.
        """
        raise NotImplemented

    @abstractmethod
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        :return: A new DataLoader instance for the training set.
        """
        raise NotImplemented

    @abstractmethod
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        :return: A new DataLoader instance for the validation set.
        """
        raise NotImplemented

    @abstractmethod
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        :return: A new DataLoader instance for the test set.
        """
        raise NotImplemented
