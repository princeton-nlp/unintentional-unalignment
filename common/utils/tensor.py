import string
from typing import Sequence, List

import numpy as np
import sklearn.metrics.pairwise
import torch
import torch.nn.functional as F


def to_numpy(tensor):
    """
    Converts tensor to numpy ndarray. Will move tensor to cpu and detach it before converison. Numpy ndarray will share memory
    of the tensor.
    :param tensor: input pytorch tensor.
    :return: numpy ndarray with shared memory of the given tensor.
    """
    return tensor.cpu().detach().numpy()


def get_softmax_predictions(logits, num_predictions=1):
    """
    Takes as input a Tensor of size(batch_size, num_classes) and returns (probabilities, predictions) tuple where
    each one is a Tensor of size (batch_size, num_predictions) containing the probabilities/indices of the predictions.
    """
    probabilities = F.softmax(logits, dim=1)
    return torch.topk(probabilities, num_predictions, dim=1, sorted=True)


def matrix_effective_rank(matrix):
    """
    Calculates the effective rank of a matrix.
    :param matrix: torch matrix of size (N, M)
    :return: Effective rank of the matrix.
    """
    svd_result = torch.svd(matrix, compute_uv=False)
    singular_values = svd_result.S
    non_zero_singular_values = singular_values[singular_values != 0]
    normalized_non_zero_singular_values = non_zero_singular_values / non_zero_singular_values.sum()

    singular_values_entropy = -(normalized_non_zero_singular_values * torch.log(normalized_non_zero_singular_values)).sum()
    return torch.exp(singular_values_entropy).item()


def cosine_similarity(tensors1, tensors2):
    """
    Given tensors1 of size (N, d) and tensors2 of shape (M, d), calculates and returns all pairwise cosine similarities in
    a tensor of size (N, M).
    """
    norms1 = tensors1.norm(p=2, dim=1, keepdim=True)
    norms2 = tensors2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(tensors1 / norms1, (tensors2 / norms2).t())


def euclidean_distance(tensors1, tensors2):
    """
    Given tensors1 of size (N, d) and tensors2 of shape (M, d), calculates and returns all pairwise euclidean distances in
    a tensor of size (N, M).
    """
    orig_device = tensors1.device

    np_tensors1 = to_numpy(tensors1)
    np_tensors2 = to_numpy(tensors2)
    dist_mat = sklearn.metrics.pairwise.euclidean_distances(np_tensors1, np_tensors2)
    return torch.from_numpy(dist_mat).to(orig_device)


def get_top_results(query_tensors, tensors, num_results=1, metric=cosine_similarity, largest=True):
    """
    Given query_tensors of size (N, d) and tensors of size (M, d) will return the top results according to a certain metric
    for each of the N input query tensors from tensors. The output is a tuple (metric_values, top_result_indices) where each
    one of them is a tensor of size (N, num_results) containing the metric values/index of the top results.
    """
    metric_matrix = metric(query_tensors, tensors)
    return torch.topk(metric_matrix, min(num_results, metric_matrix.size(1)), dim=1, largest=largest)


def get_feature_parts_top_results(query_tensors_feature_parts, tensor_parts, weights, num_results=1, metric=cosine_similarity, largest=True):
    """
    Given sequence of query tensors where each tensor in the sequence is of size (N,*) and a sequence of tensors of size (M,*) where the dimensions
    match per element in the sequences will return the top results according to a certain metric for each of the N input query tensors from tensors.
    Each part of the metric is weighted according to the given weights on the different parts of the tensors. The output is a tuple
    (metric_values, top_result_indices) where each one of them is a tensor of size (N, num_results) containing the metric values/index of the top
    results.
    """
    abs_weights_sum = sum([abs(weight) for weight in weights])
    metric_mats = [metric(query_tensors_feature_parts[i], tensor_parts[i]) * weights[i] / abs_weights_sum
                   for i in range(len(query_tensors_feature_parts))]

    total_metric_mat = sum(metric_mats)
    return torch.topk(total_metric_mat, min(num_results, total_metric_mat.size(1)), dim=1, largest=largest)


def get_many_to_one_top_results(many_to_one_query_tensors, tensors, reduce=lambda x: torch.mean(x, dim=0, keepdim=True),
                                num_results=1, metric=cosine_similarity, largest=True):
    """
    Given many_to_one_query_tensors of size (num_tensors_in_query, d) and tensors of size (M, d) will return the top results according to a certain metric
    and reduce function that combines the results from each of the query tensors. The output is a tuple (metric_values, top_result_indices) where
    each of them is a tensor of size (1, num_results) containing the final metric values/index of the top results.
    """
    metric_matrix = metric(many_to_one_query_tensors, tensors)
    reduced_results = reduce(metric_matrix)
    return torch.topk(reduced_results, min(num_results, metric_matrix.size(1)), dim=1, largest=largest)


def get_feature_parts_many_to_one_top_results(many_to_one_query_tensors_parts, tensors_parts, weights,
                                              reduce=lambda x: torch.mean(x, dim=0, keepdim=True),
                                              num_results=1, metric=cosine_similarity, largest=True):
    """
    Given many_to_one_query_tensors sequence of tensors of size (num_tensors_in_query, *) and tensors sequence of sizes (M, *)  where the dimensions
    match per element in the sequences will return the top results according to a certain metric and reduce function that combines the results from
    each of the query tensors. Each part of the metric is weighted according to the given weights on the different parts of the tensors.
    The output is a tuple (metric_values, top_result_indices) where each of them is a tensor of size (1, num_results) containing the final
    metric values/index of the top results.
    """
    abs_weights_sum = sum([abs(weight) for weight in weights])
    metric_mats = [metric(many_to_one_query_tensors_parts[i], tensors_parts[i]) * weights[i] / abs_weights_sum
                   for i in range(len(many_to_one_query_tensors_parts))]

    total_metric_mat = sum(metric_mats)
    reduced_results = reduce(total_metric_mat)
    return torch.topk(reduced_results, min(num_results, total_metric_mat.size(1)), dim=1, largest=largest)


def get_reduced_metric(query_tensors, tensors, metric, reduce=torch.mean):
    """
    Given query_tensors of size (N,d) and tensors of size (M, d) will return a tensor of size (N, 1) where the i'th entry is
    the reduced value of  all pairwise metric between the i'th query tensor to the tensors.
    """
    metric_mat = metric(query_tensors, tensors)
    return reduce(metric_mat, dim=1, keepdim=True)


def reconstruct_parafac(factors, coefficients: torch.Tensor = None) -> torch.Tensor:
    """
    Reconstructs a tensor from its CP decomposition. Each factor i is of size (d_i, r), and the tensor is created by computing the tensor
    product for the corresponding vectors in each factor and summing all r outer products.
    :param factors: List of tensors of size (d_i, r).
    :param coefficients: optional coefficients tensor of size (r,), used to multiply each summand in the CP decomposition.
    :return: Tensor of size (d_1,...,d_n)
    """
    ndims = len(factors)
    request = ''
    for temp_dim in range(ndims):
        request += string.ascii_lowercase[temp_dim] + "z,"

    if coefficients is not None:
        factors = factors + [coefficients]
        request += "z,"

    request = request[:-1] + "->" + string.ascii_lowercase[:ndims]
    return torch.einsum(request, *factors)


def compute_parafac_components(factors, coefficients: torch.Tensor = None) -> torch.Tensor:
    """
    Reconstructs the CP decomposition components. Each factor i is of size (d_i, r), and the output is created by computing the tensor
    product for the corresponding vectors in each factor and stacking them.
    :param factors: List of tensors of size (d_i, r).
    :param coefficients: optional coefficients tensor of size (r,), used to multiply each component in the CP decomposition.
    :return: Tensor of size (r, d_1,...,d_n)
    """
    ndims = len(factors)
    request = ''
    for temp_dim in range(ndims):
        request += string.ascii_lowercase[temp_dim] + "z,"

    if coefficients is not None:
        factors = factors + [coefficients]
        request += "z,"

    request = request[:-1] + "->" + "z" + string.ascii_lowercase[:ndims]
    return torch.einsum(request, *factors)


def tensor_product(first_tensor: torch.Tensor, second_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the tensor product between the given tensors.
    :param first_tensor: Tensor of size (d_1,...,d_N)
    :param second_tensor: Tensor of size (s_1,...,s_M)
    :return: Tensor of size (d_1,...,d_N,s_1,...,s_M) that is the tensor product of the input tensors.
    """
    first_tensor_ndims = len(first_tensor.size())
    second_tensor_ndims = len(second_tensor.size())

    request = ""
    request += string.ascii_lowercase[: first_tensor_ndims] + ","
    request += string.ascii_lowercase[first_tensor_ndims: first_tensor_ndims + second_tensor_ndims]
    request += "->" + string.ascii_lowercase[:first_tensor_ndims + second_tensor_ndims]
    return torch.einsum(request, first_tensor, second_tensor)


def matricize(tensor: torch.Tensor, row_modes: Sequence[int]):
    """
    Returns the matricization for a given tensor and row modes.
    """
    col_modes = [j for j in range(len(tensor.shape)) if j not in row_modes]
    permute_indices = list(row_modes) + col_modes

    row_dim = int(np.prod([tensor.shape[i] for i in row_modes]))
    col_dim = int(np.prod([tensor.shape[i] for i in col_modes]))

    matricized_tensor = tensor.permute(*permute_indices).reshape(row_dim, col_dim)
    return matricized_tensor


def dematricize(matrix: torch.Tensor, row_modes: List[int], row_modes_dims: List[int],
                col_modes: List[int], col_modes_dims: List[int]):
    """
    Returns the dematricization of a matrix (reshaping it to a tensor) according to the given dimensions of modes corresponding to rows and columns
    in the matrix. This operation reverts the matricization operation done with the same modes.
    :param matrix: matrix to reshape into a tensor.
    :param row_modes: indices of modes in the resulting tensor that correspond to the rows in the matrix.
    :param row_modes_dims: dimensions of row modes.
    :param col_modes: indices of modes in the resulting tensor that correspond to the columns in the matrix.
    :param col_modes_dims: dimensions of column modes.
    """
    mode_dims = row_modes_dims + col_modes_dims
    tensorized_matrix = matrix.reshape(*mode_dims)

    mode_indices = row_modes + col_modes
    permutation = [0] * len(mode_indices)
    for i in range(len(mode_indices)):
        permutation[mode_indices[i]] = i

    return tensorized_matrix.permute(*permutation)


def create_tensor_with_cp_rank(num_dim_per_mode, cp_rank: int, init_mean: float = 0.0, init_std: float = 1.,
                               fro_norm: float = 1, symmetric: bool = False) -> torch.Tensor:
    """
    Creates a tensor with the given CP rank.
    :param num_dim_per_mode: list of dimensions per mode
    :param cp_rank: desired CP rank (will determine number size of factors per mode). If non-positive, will randomly sample tensor entries directly
    :param init_mean: Mean of Gaussian initialization
    :param init_std: Standard deviation of Gaussian initialization
    :param fro_norm: Frobenius norm of tensor (the tensor is normalized at the end to this norm). If <= 0, will not normalize. Default is 1.
    :param symmetric: if True will return a super symmetric tensor
    :return: Tensor of the given CP rank and Frobenius norm
    """
    if symmetric:
        return create_super_symmetric_tensor_with_cp_rank(num_dim_per_mode[0], len(num_dim_per_mode), cp_rank, init_mean, init_std, fro_norm)

    if cp_rank <= 0:
        tensor = torch.randn(*num_dim_per_mode) * init_std + init_mean
    else:
        factors = []
        for dim in num_dim_per_mode:
            factor = torch.randn(dim, cp_rank) * init_std + init_mean
            factors.append(factor)

        tensor = reconstruct_parafac(factors)

    if fro_norm > 0:
        tensor = (tensor / torch.norm(tensor, p="fro")) * fro_norm

    return tensor


def create_super_symmetric_tensor_with_cp_rank(modes_dim: int, order: int, cp_rank: int, init_mean: float = 0.0, init_std: float = 1.,
                                               fro_norm: float = 1) -> torch.Tensor:
    """
    Creates a super symmetric tensor with the given CP rank.
    :param modes_dim: number of dimensions in each mode
    :param order: order of the tensor to create
    :param cp_rank: desired CP rank (will determine number size of factors per mode)
    :param init_mean: Mean of Gaussian initialization
    :param init_std: Standard deviation of Gaussian initialization
    :param fro_norm: Frobenius norm of tensor (the tensor is normalized at the end to this norm). If <= 0, will not normalize. Default is 1.
    :return: super symmetric Tensor of the given CP rank and Frobenius norm
    """
    factor = torch.randn(modes_dim, cp_rank) * init_std + init_mean
    tensor = reconstruct_parafac([factor] * order)

    if fro_norm > 0:
        tensor = (tensor / torch.norm(tensor, p="fro")) * fro_norm

    return tensor


def reconstruct_index(factors, indices):
    """
    Reconstructs the given indices tensor from a CP decomposition.
    :param factors: List of tensors of size (d_i, r).
    :param indices: Tensor of size (*, n) with each row corresponding to an index tuple of the tensor.
    :return: Tensor of size (*, ) with the CP decomposition values at the given indices.
    """
    new_factors = [factor[index] for factor, index in zip(factors, indices.t())]
    return torch.einsum(('za,' * len(new_factors))[:-1] + '->z', *new_factors)


def convert_tensor_to_one_hot(tensor: torch.Tensor, num_options: int = -1):
    """
    Converts an input tensor with indices to a corresponding one hot representation of each index.
    :param tensor: Tensor of size (d_1,...,d_N) with integers.
    :param num_options: Number of optional integers in the given tensor. This is the dimension of the one hot representation. If none given, takes
    the maximal integer in the given tensor as the size of the representation.
    :return:Float Tensor of size (d_1,...,d_N,d), where d is the size of the one hot representation.
    """
    if tensor.numel() == 0:
        return tensor

    flattened_tensor = tensor.flatten().long()

    num_options = max(num_options, (flattened_tensor.max() + 1).int())
    one_hot_tensor_result = torch.zeros(flattened_tensor.numel(), num_options, device=tensor.device)
    one_hot_tensor_result[torch.arange(flattened_tensor.numel()), flattened_tensor] = 1
    one_hot_tensor_result = one_hot_tensor_result.reshape(*tensor.shape, -1)

    return one_hot_tensor_result.float()


def convert_one_hot_to_int(tensor: torch.Tensor):
    """
    Converts one-hot encodings to the corresponding integer representations. For input tensor of shape (d_1,..., d_N, d) the output is of shape
    (d_1,...,d_N), where each entry is the index of the maximal value in the last axis.
    """
    return tensor.argmax(dim=-1)
