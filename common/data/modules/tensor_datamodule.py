import torch.utils.data
from sklearn.model_selection import train_test_split

from .datamodule import DataModule
from ..loaders.fast_tensor_dataloader import FastTensorDataLoader


class TensorDataModule(DataModule):

    def __init__(self, train_dataset_path: str, val_dataset_path: str, test_dataset_path: str = None,
                 num_train_samples: int = -1, batch_size: int = 32, split_random_state: int = -1):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.num_train_samples = num_train_samples
        self.batch_size = batch_size
        self.split_random_state = split_random_state

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        self.train_dataset = torch.utils.data.TensorDataset(torch.load(self.train_dataset_path))
        self.val_dataset = torch.utils.data.TensorDataset(torch.load(self.val_dataset_path))
        self.test_dataset = self.val_dataset if not self.test_dataset_path else torch.utils.data.TensorDataset(torch.load(self.test_dataset_path))

        if 0 < self.num_train_samples < len(self.train_dataset):
            self.train_dataset = self.__subsample_dataset(self.train_dataset, self.num_train_samples)

    def __subsample_dataset(self, dataset: torch.utils.data.Dataset, num_samples: int):
        train_indices, _ = train_test_split(torch.arange(len(dataset)), train_size=num_samples,
                                            random_state=self.split_random_state if self.split_random_state > 0 else None)
        subsampled_train_dataset_tensors = [tensor[train_indices] for tensor in self.train_dataset.tensors]
        return torch.utils.data.TensorDataset(*subsampled_train_dataset_tensors)

    def train_dataloader(self):
        batch_size = self.batch_size if self.batch_size > 0 else len(self.train_dataset)
        return FastTensorDataLoader(self.train_dataset.tensors, batch_size=batch_size, shuffle=True)

    def val_dataloader(self):
        batch_size = self.batch_size if self.batch_size > 0 else len(self.val_dataset)
        return FastTensorDataLoader(self.val_dataset.tensors, batch_size=batch_size, shuffle=False)

    def test_dataloader(self):
        batch_size = self.batch_size if self.batch_size > 0 else len(self.test_dataset)

        return FastTensorDataLoader(self.test_dataset.tensors, batch_size=batch_size, shuffle=False)
