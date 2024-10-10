from .evaluator import MetricsEvaluator, TrainEvaluator
from ..metrics import DummyAveragedMetric, MetricInfo


class TrainBatchOutputEvaluator(TrainEvaluator):
    """
    Train evaluator for tracking metrics that are already calculated during the training batch. Takes the values from the given
    output and stores them in tracked values.
    """

    def __init__(self, metric_names, metric_tags=None):
        self.metric_names = metric_names
        self.metric_tags = metric_tags if metric_tags is not None else metric_names

        self.metric_infos = {metric_names[i]: MetricInfo(metric_names[i], DummyAveragedMetric(), self.metric_tags[i])
                             for i in range(len(metric_names))}
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def evaluate_batch(self, output):
        num_samples = 1 if "num_samples" not in output else output["num_samples"]
        metric_values = {}
        for name, metric in self.metrics.items():
            value = output[name]
            metric(value, num_samples)
            self.tracked_values[name].add_batch_value(value)
            metric_values[name] = value

        return metric_values
