from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn

from .metric import AveragedMetric
from ...utils import module as module_utils


class ParametersMetric(AveragedMetric, ABC):
    """
    Abstract class for metrics that are a function of the concatenation of parameters.
    """

    def __init__(self, exclude: Sequence[type] = None, include_only: Sequence[type] = None, exclude_by_name_part: Sequence[str] = None):
        """
        @param exclude: sequence of module types to exclude.
        @param include_only: sequence of module types to include only. If None, then will include by default all layer types.
        @param exclude_by_name_part: sequence of strings to exclude parameters which include one of the given names in as part of their name.
        """
        super().__init__()
        self.exclude = exclude
        self.include_only = include_only
        self.exclude_by_name_part = exclude_by_name_part

    def _calc_metric(self, module: nn.Module):
        """
        :param module: PyTorch module.
        :return: (metric value, 1)
        """
        params = list(module_utils.get_parameters_iter(module,
                                                       exclude=self.exclude,
                                                       include_only=self.include_only,
                                                       exclude_by_name_part=self.exclude_by_name_part))
        flattened_params_vector = torch.cat([param.view(-1) for param in params])

        return self._compute_metric_over_params_vector(flattened_params_vector), 1

    @abstractmethod
    def _compute_metric_over_params_vector(self, flattened_params_vector: torch.Tensor):
        """
        :param flattened_params_vector: all relevant module parameters concatenated as a vector.
        :return: metric value.
        """
        raise NotImplementedError


class ParameterValueMean(ParametersMetric):
    """
    Mean of parameter values metric. Allows to compute mean only for specific types of layers.
    """

    def _compute_metric_over_params_vector(self, flattened_params_vector: torch.Tensor):
        return flattened_params_vector.mean().item()


class ParameterValueSTD(ParametersMetric):
    """
    Standard deviation of parameter values metric. Allows to compute mean only for specific types of layers.
    """

    def _compute_metric_over_params_vector(self, flattened_params_vector: torch.Tensor):
        return flattened_params_vector.std().item()


class ParameterValueQuantile(ParametersMetric):
    """
    Quantile parameter value metric. E.g. Allows to compute the median parameter value.
    """

    def __init__(self, quantile: float = 0.5, exclude: Sequence[type] = None, include_only: Sequence[type] = None,
                 exclude_by_name_part: Sequence[str] = None):
        """
        @param quantile: quantile value of parameters to return.
        @param exclude: sequence of module types to exclude.
        @param include_only: sequence of module types to include only. If None, then will include by default all layer types.
        @param exclude_by_name_part: sequence of strings to exclude parameters which include one of the given names in as part of their name.
        """
        super().__init__(exclude=exclude, include_only=include_only, exclude_by_name_part=exclude_by_name_part)
        self.quantile = quantile

    def _compute_metric_over_params_vector(self, flattened_params_vector: torch.Tensor):
        return torch.quantile(flattened_params_vector, q=self.quantile).item()


class ParameterAbsoluteValueMean(ParametersMetric):
    """
    Mean of parameter absolute values metric. Allows to compute mean only for specific types of layers.
    """

    def _compute_metric_over_params_vector(self, flattened_params_vector: torch.Tensor):
        flattened_abs_params_vector = torch.abs(flattened_params_vector)
        return flattened_abs_params_vector.mean().item()


class ParameterAbsoluteValueSTD(ParametersMetric):
    """
    Standard deviation of parameter absolute values metric. Allows to compute mean only for specific types of layers.
    """

    def _compute_metric_over_params_vector(self, flattened_params_vector: torch.Tensor):
        flattened_abs_params_vector = torch.abs(flattened_params_vector)
        return flattened_abs_params_vector.std().item()


class ParameterAbsoluteValueQuantile(ParametersMetric):
    """
    Quantile absolute parameter value metric. E.g. Allows to compute the median parameter absolute value.
    """

    def __init__(self, quantile: float = 0.5, exclude: Sequence[type] = None, include_only: Sequence[type] = None,
                 exclude_by_name_part: Sequence[str] = None):
        """
        @param quantile: quantile value of parameters to return.
        @param exclude: sequence of module types to exclude.
        @param include_only: sequence of module types to include only. If None, then will include by default all layer types.
        @param exclude_by_name_part: sequence of strings to exclude parameters which include one of the given names in as part of their name.
        """
        super().__init__(exclude=exclude, include_only=include_only, exclude_by_name_part=exclude_by_name_part)
        self.quantile = quantile

    def _compute_metric_over_params_vector(self, flattened_params_vector: torch.Tensor):
        flattened_abs_params_vector = torch.abs(flattened_params_vector)
        return torch.quantile(flattened_abs_params_vector, q=self.quantile).item()


class ParameterAbsoluteValueMax(ParametersMetric):
    """
    Max of parameter absolute values metric. Allows to compute max only for specific types of layers.
    """

    def _compute_metric_over_params_vector(self, flattened_params_vector: torch.Tensor):
        flattened_abs_params_vector = torch.abs(flattened_params_vector)
        return flattened_abs_params_vector.max().item()
