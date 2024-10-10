from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluator import MetricsEvaluator, Evaluator, TrainEvaluator
from .. import metrics as metrics
from ...utils import module as module_utils


class SupervisedTrainEvaluator(TrainEvaluator):
    """
    Train evaluator for regular supervised task of predicting y given x (classification or regression).
    """

    def __init__(self, metric_info_seq: Sequence[metrics.MetricInfo] = None):
        self.metric_infos = metrics.metric_info_seq_to_dict(metric_info_seq) if metric_info_seq is not None else {}
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def evaluate_batch(self, output):
        y_pred = output["y_pred"]
        y = output["y"]

        metric_values = {}
        for name, metric in self.metrics.items():
            value = metric(y_pred, y)
            self.tracked_values[name].add_batch_value(value)
            metric_values[name] = value

        return metric_values


class SupervisedValidationEvaluator(Evaluator):
    """
    Validation evaluator for regular supervised task of predicting y given x (classification or regression).
    """

    def __init__(self, model: nn.Module, data_loader: DataLoader, metric_info_seq: Sequence[metrics.MetricInfo] = None,
                 device=torch.device("cpu")):
        self.metric_infos = metrics.metric_info_seq_to_dict(metric_info_seq) if metric_info_seq is not None else {}
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

        self.model = model
        self.data_loader = data_loader
        self.device = device

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def evaluate(self):
        with torch.no_grad():
            self.model.to(self.device)
            for x, y in self.data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)

                for name, metric in self.metrics.items():
                    value = metric(y_pred, y)
                    self.tracked_values[name].add_batch_value(value)

            eval_metric_values = {name: metric.current_value() for name, metric in self.metrics.items()}
            return eval_metric_values
