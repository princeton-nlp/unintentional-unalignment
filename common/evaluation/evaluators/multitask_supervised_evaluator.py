import torch

from .evaluator import MetricsEvaluator, Evaluator, TrainEvaluator
from .. import metrics as metrics
from ...utils import module as module_utils


def create_task_metric_name(task_name: str, metric_name: str):
    """
    Creates the tracked value name for the given task and metric names.
    :param task_name: task name.
    :param metric_name: metric name.
    :return: name identifier for the metric.
    """
    return f"{task_name}_{metric_name}"


def _create_metric_infos(by_task_metric_infos):
    metric_infos = {}
    for task_name, metric_info_dict in by_task_metric_infos.items():
        for metric_name, metric_info in metric_info_dict.items():
            name = create_task_metric_name(task_name, metric_name)
            metric_infos[name] = metric_info

    return metric_infos


def _create_metrics(by_task_metrics):
    metrics_dict = {}
    for task_name, metrics_dict in by_task_metrics.items():
        for metric_name, metric in metrics_dict.items():
            name = create_task_metric_name(task_name, metric_name)
            metrics_dict[name] = metric

    return metrics_dict


class MultitaskSupervisedTrainEvaluator(TrainEvaluator):
    """
    Train evaluator for multitask supervised tasks of predicting multiple outputs given x (classification or regression).
    """

    def __init__(self, by_task_metric_info_seq=None):
        if by_task_metric_info_seq is None:
            self.by_task_metric_infos = {}
        else:
            self.by_task_metric_infos = {task_name: metrics.metric_info_seq_to_dict(metric_info_seq)
                                         for task_name, metric_info_seq in by_task_metric_info_seq.items()}

        self.by_task_metrics = {task_name: metrics.get_metric_dict_from_metric_info_dict(metric_info)
                                for task_name, metric_info in self.by_task_metric_infos}

        self.metric_infos = _create_metric_infos(self.by_task_metric_infos)
        self.metrics = _create_metrics(self.by_task_metrics)
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def evaluate_batch(self, output):
        by_task_y_pred = output["by_task_y_pred"]
        by_task_y = output["by_task_y"]

        metric_values = {}
        for task_name, metrics_dict in self.by_task_metrics.items():
            if not metrics_dict:
                continue

            y_pred = by_task_y_pred[task_name]
            y = by_task_y[task_name]

            for metric_name, metric in metrics_dict.items():
                value = metric(y_pred, y)

                full_metric_name = create_task_metric_name(task_name, metric_name)
                self.tracked_values[full_metric_name].add_batch_value(value)
                metric_values[full_metric_name] = value

        return metric_values


class MultitaskSupervisedValidationEvaluator(Evaluator):
    """
    Validation evaluator for multitask supervised tasks of predicting multiple outputs given x (classification or regression).
    """

    def __init__(self, model, data_loader, by_task_metric_info_seq=None, device=torch.device("cpu")):
        if by_task_metric_info_seq is None:
            self.by_task_metric_infos = {}
        else:
            self.by_task_metric_infos = {task_name: metrics.metric_info_seq_to_dict(metric_info_seq)
                                         for task_name, metric_info_seq in by_task_metric_info_seq.items()}

        self.by_task_metrics = {task_name: metrics.get_metric_dict_from_metric_info_dict(metric_info)
                                for task_name, metric_info in self.by_task_metric_infos}

        self.metric_infos = _create_metric_infos(self.by_task_metric_infos)
        self.metrics = _create_metrics(self.by_task_metrics)
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
            for x, by_task_y in self.data_loader:
                x = x.to(self.device)
                by_task_y = {task_name: y.to(self.device) for task_name, y in by_task_y.items()}

                by_task_y_pred = self.model(x)

                for task_name, metrics_dict in self.by_task_metrics.items():
                    if not metrics_dict:
                        continue

                    y_pred = by_task_y_pred[task_name]
                    y = by_task_y[task_name]

                    for metric_name, metric in metrics_dict.items():
                        value = metric(y_pred, y)
                        full_metric_name = create_task_metric_name(task_name, metric_name)
                        self.tracked_values[full_metric_name].add_batch_value(value)

            eval_metric_values = self.__get_current_metric_values()
            return eval_metric_values

    def __get_current_metric_values(self):
        metric_values = {}
        for task_name, metrics_dict in self.by_task_metrics.items():
            for metric_name, metric in metrics_dict.items():
                metric_values[create_task_metric_name(task_name, metric_name)] = metric.current_value()

        return metric_values
