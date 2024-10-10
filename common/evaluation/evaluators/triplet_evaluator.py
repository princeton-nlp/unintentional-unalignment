import torch

from .evaluator import MetricsEvaluator, Evaluator, TrainEvaluator
from .. import metrics as metrics
from ...utils import module as module_utils


class TripletTrainEvaluator(TrainEvaluator):
    """
    Train evaluator for triplet ranking task. Supports metrics that receives query, positive, negative batches.
    """

    def __init__(self, metric_info_seq=None):
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
        query = output["query"]
        positive = output["positive"]
        negative = output["negative"]

        metric_values = {}
        for name, metric in self.metrics.items():
            value = metric(query, positive, negative)
            self.tracked_values[name].add_batch_value(value)
            metric_values[name] = value

        return metric_values


class TripletValidationEvaluator(Evaluator):
    """
    Validation evaluator for triplet ranking task. Supports metrics that receives query, positive, negative batches.
    """

    def __init__(self, model, val_triplet_data_loader, metric_info_seq=None, device=torch.device("cpu")):
        self.metric_infos = metrics.metric_info_seq_to_dict(metric_info_seq) if metric_info_seq is not None else {}
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

        self.model = model
        self.val_triplet_data_loader = val_triplet_data_loader
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
            for query, positive, negative in self.val_triplet_data_loader:
                query = query.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                query = self.model(query)
                positive = self.model(positive)
                negative = self.model(negative)

                for name, metric in self.metrics.items():
                    value = metric(query, positive, negative)
                    self.tracked_values[name].add_batch_value(value)

            eval_metric_values = {name: metric.current_value() for name, metric in self.metrics.items()}
            return eval_metric_values
