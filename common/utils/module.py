from typing import Callable
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from . import logging as logging_utils


class Flatten(nn.Module):
    """
    Flattens the input into a tensor of size (batch_size, num_elements_in_tensor).
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Normalize(nn.Module):
    """
    Normalizes by dividing by the norm of the input tensors.
    """

    def __init__(self, p=2, dim=1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)


class PassthroughLayer(nn.Module):
    """
    Passthrough layer that just returns the input.
    """

    def forward(self, x):
        return x


class Permute(nn.Module):
    """
    Permutes values of input according to a certain dimension.
    """

    def __init__(self, dim: int, create_perm_func: Callable[[int], torch.tensor]):
        """
        :param dim: dimension to permute.
        :param create_perm_func: function that takes as input the dimension size and returns a tensor of indices (dtype Long) according to which
        the input will be permuted.
        """
        super().__init__()
        self.dim = dim
        self.create_perm_func = create_perm_func

    def forward(self, x):
        perm = self.create_perm_func(x.shape[self.dim])
        return torch.index_select(x, dim=self.dim, index=perm)


class NegativeSamplingLinear(nn.Module):
    """
    Negative sampling linear layer wrapper that runs the linear layer only for a sample of the negative outputs.
    Used for subsampling before approximating cross entropy loss.
    """

    def __init__(self, linear_layer, negative_sample_ratio=1, sampler=None, normalize_linear_layer=False, reuse_negative_samples=False):
        """
        :param linear_layer: Linear layer to wrap.
        :param negative_sample_ratio: ratio of negative examples out of all possible examples to use per input.
        :param sampler: callable object that given the correct input class index y returns a sequence of sampled indices to use.
        :param normalize_linear_layer: flag whether to normalize linear layer weights to unit vectors during forward pass.
        :param reuse_negative_samples: flag whether to reuse the same negatives for all samples in batch.
        """
        super().__init__()

        if sampler is not None and negative_sample_ratio != 1:
            raise ValueError("sampler option is mutually exclusive with subsample ratio")

        self.linear_layer = linear_layer
        self.negative_sample_ratio = negative_sample_ratio
        self.sampler = sampler
        self.normalize_linear_layer = normalize_linear_layer
        self.reuse_negatives = reuse_negative_samples

    def forward(self, x):
        if self.normalize_linear_layer:
            self.linear_layer.weight.data = F.normalize(self.linear_layer.weight.data, p=2, dim=1)
        return self.linear_layer(x)

    def negative_sample_forward(self, x, y):
        if self.normalize_linear_layer:
            self.linear_layer.weight.data = F.normalize(self.linear_layer.weight.data, p=2, dim=1)

        if self.sampler is None and self.negative_sample_ratio == 1:
            return self.linear_layer(x), y

        if not self.reuse_negatives:
            # when subsampling the correct target will always be the first (index 0)
            y_zeros = torch.zeros_like(y)
            return self.__with_negative_samples_mm(x, y), y_zeros
        else:
            return self.__reuse_negatives_mm(x, y)

    def __with_negative_samples_mm(self, x, y):
        batch_samples_softmax_mat = []
        for i in range(len(y)):
            cur_label = y[i]
            positive_sample = self.linear_layer.weight[cur_label: cur_label + 1]

            neg_samples_indices = self.__get_negative_samples_indices(cur_label)
            neg_samples = self.linear_layer.weight[neg_samples_indices]
            negative_sampled_linear_mat = torch.cat([positive_sample, neg_samples]).t()
            batch_samples_softmax_mat.append(negative_sampled_linear_mat)

        batch_samples_softmax_mat = torch.stack(batch_samples_softmax_mat)
        return torch.bmm(x.unsqueeze(1), batch_samples_softmax_mat).squeeze(1)

    def __get_negative_samples_indices(self, label):
        if self.sampler is not None:
            return self.sampler(label)

        num_labels = self.linear_layer.weight.size(0)
        neg_samples_options = [j for j in range(num_labels) if j != label]
        num_neg_samples = int(self.negative_sample_ratio * num_labels)
        return np.random.choice(neg_samples_options, num_neg_samples, replace=False)

    def __reuse_negatives_mm(self, x, y):
        new_y = torch.zeros_like(y)
        for i in range(len(new_y)):
            new_y[i] = i

        curr_batch_samples = self.linear_layer.weight[y]
        neg_samples_indices = self.__get_negative_samples_indices(y[0])
        neg_samples = self.linear_layer.weight[neg_samples_indices]
        batch_samples_softmax_mat = torch.cat([curr_batch_samples, neg_samples])

        return torch.mm(x, batch_samples_softmax_mat.t()), new_y


def get_number_of_parameters(module: nn.Module) -> int:
    """
    Returns the number of parameters in the module.
    :param module: PyTorch Module.
    :return: Number of parameters in the module.
    """
    return sum(p.numel() for p in module.parameters())


def get_use_gpu(disable_gpu: bool = False) -> bool:
    """
    Returns true if cuda is available and no explicit disable cuda flag given.
    """
    return torch.cuda.is_available() and not disable_gpu


def get_device(disable_gpu: bool = False, cuda_id: int = 0):
    """
    Returns a gpu cuda device if available and cpu device otherwise.
    """
    if get_use_gpu(disable_gpu):
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def compute_grad_norm(module: nn.Module, detach: bool = False):
    """
    Returns the Euclidean norm of the current gradients.
    :param module: pytorch module.
    :param detach: whether to detach gradients from computation graph before computing the norm.
    """
    grads = []
    for param in module.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().flatten() if detach else param.grad.flatten())

    return torch.cat(grads).norm()


def set_requires_grad(module: nn.Module, requires_grad: bool):
    """
    Sets the requires grad flag for all of the modules parameters.
    :param module: pytorch module.
    :param requires_grad: requires grad flag value.
    """
    for param in module.parameters():
        param.requires_grad = requires_grad


def get_parameters_iter(module: nn.Module, exclude: Sequence[type] = None, include_only: Sequence[type] = None,
                        exclude_by_name_part: Sequence[str] = None) -> Sequence[nn.Parameter]:
    """
    Gets an iterator that iterates over the module parameters.
    @param module: module to get an iterator over its parameters.
    @param exclude: sequence of module types to exclude.
    @param include_only: sequence of module types to include only. If None, then will include by default all layer types.
    @param exclude_by_name_part: sequence of strings to exclude parameters which include one of the given names in as part of their name.
    @return iterator over module parameters.
    """
    exclude_by_name_part = exclude_by_name_part if exclude_by_name_part is not None else []
    for child_module in module.modules():
        if include_only is not None and child_module.__class__ not in include_only:
            continue

        if exclude is not None and child_module.__class__ in exclude:
            continue

        for name, param in child_module.named_parameters(recurse=False):
            exclude_param = any([exclude_name_part in name for exclude_name_part in exclude_by_name_part])
            if exclude_param:
                continue

            yield param


def create_sequential_model_without_top(model: nn.Module, num_top_layers: int = 1) -> nn.Module:
    """
    Creates and returns a model that is the same as the input model with the number of top layers removed. The model
    share their state.
    :param model: input pytorch module.
    :param num_top_layers: number of top layers to remove from the new model.
    :return: new model with same state as the input model with the number of top layers removed.
    """
    model_children_without_top_layers = list(model.children())[:-num_top_layers]
    return nn.Sequential(*model_children_without_top_layers, Flatten())


def predict_in_batches(model: nn.Module, dataset: torch.utils.data.Dataset, batch_size: int = 64,
                       log_every_num_batches: int = -1, device=torch.device("cpu")) -> torch.Tensor:
    """
    Runs model in batches on dataset and returns a tensor with the concatenated results.
    """
    batch_predictions = []
    for i, batch_prediction in enumerate(iter_batch_predictions(model, dataset, batch_size=batch_size, device=device)):
        batch_predictions.append(batch_prediction)

        if log_every_num_batches != -1 and (i + 1) % log_every_num_batches == 0:
            logging_utils.info(f"Finished running prediction on batch number {i + 1}")

    return torch.cat(batch_predictions)


def iter_batch_predictions(model: nn.Module, dataset: torch.utils.data.Dataset, batch_size: int = 64, device=torch.device("cpu")) -> torch.Tensor:
    """
    Generator for predicting with the given model the inputs in batches.
    """
    model = model.to(device)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for input_batch in data_loader:
        input_batch = input_batch.to(device)
        yield model(input_batch)


def export_to_onnx(model: nn.Module, input_sizes: Sequence[int], output_path: str, input_names: Sequence[str] = None,
                   output_names: Sequence[str] = None, export_params: bool = True, verbose: bool = True, device=torch.device("cpu")):
    """
    Exports model graph in onnx format.
    :param model: PyTorch module.
    :param input_sizes: sequence of input sizes the model receives.
    :param output_path: path to save the onnx file.
    :param input_names: optional list of names for the inputs.
    :param output_names: optional list of names for the outputs.
    :param export_params: flag whether to store param weights in the model file.
    :param verbose: if specified, we will print out a debug description of the trace being exported.
    :param device: device to run inputs through the model.
    """
    model = model.to(device)
    dummy_inputs = tuple([torch.randn(input_size, device=device) for input_size in input_sizes])
    torch.onnx.export(model, dummy_inputs, output_path, input_names=input_names, output_names=output_names, export_params=export_params,
                      verbose=verbose)
