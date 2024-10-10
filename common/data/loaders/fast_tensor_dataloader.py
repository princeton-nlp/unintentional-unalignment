import torch
from torch import Tensor


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors: Tensor, batch_size: int = -1, shuffle: bool = False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load. If <= 0, will use the size of the whole dataset.
        :param shuffle: if True, shuffle the data whenever an iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size if batch_size > 0 else self.dataset_len
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1

        self.n_batches = n_batches

    def __iter__(self):
        return FastTensorDataLoaderIter(self)

    def __len__(self):
        return self.n_batches


class FastTensorDataLoaderIter:
    """
    Iterator class for FastTensorDataLoader.
    """

    def __init__(self, fast_tensor_dataloader: FastTensorDataLoader):
        self.tensors = fast_tensor_dataloader.tensors
        self.batch_size = fast_tensor_dataloader.batch_size
        self.shuffle = fast_tensor_dataloader.shuffle
        self.dataset_len = self.tensors[0].shape[0]

        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]

        self.current_sample_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_sample_index >= self.dataset_len:
            raise StopIteration

        batch = tuple(t[self.current_sample_index:self.current_sample_index + self.batch_size] for t in self.tensors)
        self.current_sample_index += self.batch_size
        return batch
