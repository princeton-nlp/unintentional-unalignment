from abc import ABC, abstractmethod
from typing import Dict, Sequence

from ..metrics import MetricInfo, Metric
from ...serialization.torch_serializable import TorchSerializable
from ...train.tracked_value import TrackedValue


class MetricsEvaluator(TorchSerializable, ABC):
    """
    Parent abstract class for a metric evaluator. Defines functionality for evaluating and accumulating metric values.
    """

    @staticmethod
    def create_tracked_values_for_metrics(metric_infos: Dict[str, MetricInfo]):
        """
        Creates tracked values for the evaluator metrics.
        :param metric_infos: Dict of metric name to MetricInfo to create TrackedValues for.
        """
        tracked_values = {}
        for metric_name, metric_info in metric_infos.items():
            tracked_value = TrackedValue(name=metric_name, save_epoch_values=metric_info.save_epoch_values,
                                         num_per_epoch_batch_histories_to_save=metric_info.num_per_epoch_batch_histories_to_save,
                                         is_scalar=metric_info.is_scalar)
            tracked_values[tracked_value.name] = tracked_value

        return tracked_values

    @abstractmethod
    def get_metric_infos(self) -> Dict[str, MetricInfo]:
        """
        :return: Dict of MetricInfo objects where the key is the metric name.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self) -> Dict[str, Metric]:
        """
        :return: Dict of Metric objects where the key is the metric name.
        """
        raise NotImplementedError

    @abstractmethod
    def get_tracked_values(self) -> Dict[str, TrackedValue]:
        """
        :return: Dict of TrackedValue objects for the evaluator metrics. The key is the metric name.
        """
        raise NotImplementedError

    def get_metric_infos_with_history(self) -> Dict[str, MetricInfo]:
        """
        :return: Dict of MetricInfo objects for the evaluator metrics, that save history for the metric values, where the key is
        the metric name.
        """
        return {name: metric_info for name, metric_info in self.get_metric_infos().items() if metric_info.save_epoch_values}

    def get_tracked_values_with_history(self) -> Dict[str, TrackedValue]:
        """
        :return: Dict of TrackedValue objects for the evaluator metrics, that save history of the metric values,
        """
        return {name: tracked_value for name, tracked_value in self.get_tracked_values().items() if tracked_value.save_epoch_values}

    def epoch_start(self, epoch_num: int):
        """
        Calls epoch start for all tracked values. Should be called at the start of each train or validation epoch stage.
        :param epoch_num: epoch number.
        """
        metrics_dict = self.get_metrics()
        tracked_values = self.get_tracked_values()

        for name, metric in metrics_dict.items():
            metric.reset_current_epoch_values()
            tracked_values[name].epoch_start(epoch_num)

    def epoch_end(self, epoch_num: int):
        """
        Calls epoch end for all tracked values. Should be called at the end of each train or validation epoch stage.
        :param epoch_num: epoch number.
        """
        metrics_dict = self.get_metrics()
        tracked_values = self.get_tracked_values()

        for name, metric in metrics_dict.items():
            if metric.has_epoch_metric_to_update():
                metric_value = metric.current_value()
                tracked_values[name].epoch_end(metric_value, epoch_num)

    def state_dict(self) -> dict:
        return {name: tracked_value.state_dict() for name, tracked_value in self.get_tracked_values().items()}

    def load_state_dict(self, state_dict: dict):
        for name, tracked_value in self.get_tracked_values().items():
            if name in state_dict:
                tracked_value.load_state_dict(state_dict[name])


class Evaluator(MetricsEvaluator, ABC):
    """
    Evaluator abstract class. Used to evaluate metrics for a model. Subclasses should register their metrics to the metrics dict for automatic
    serialization support.
    """

    @abstractmethod
    def evaluate(self) -> dict:
        """
        Evaluates model updating metrics and returning calculated metrics.
        :return: calculated metric values.
        """
        raise NotImplementedError


class TrainEvaluator(MetricsEvaluator, ABC):
    """
    Train evaluator abstract class. Used to evaluate metrics for training phase. Subclasses should register their metrics to the metrics dict for
    automatic serialization support.
    """

    @abstractmethod
    def evaluate_batch(self, output) -> dict:
        """
        Evaluates model updating metrics using the given model outputs on a train batch and returning calculated metrics.
        :param output: train phase output.
        :return: calculated batch metric values.
        """
        raise NotImplementedError


class VoidEvaluator(Evaluator, TrainEvaluator):
    """
    Void evaluator. Does nothing.
    """

    def get_metric_infos(self):
        return {}

    def get_metrics(self):
        return {}

    def get_tracked_values(self) -> Dict[str, TrackedValue]:
        return {}

    def evaluate(self):
        return {}

    def evaluate_batch(self, output):
        return {}


def _aggregate_dictionaries(dictionary_seq):
    aggregated_dict = {}
    for dictionary in dictionary_seq:
        _verify_no_name_collision(aggregated_dict, dictionary)
        aggregated_dict.update(dictionary)

    return aggregated_dict


def _verify_no_name_collision(first_dict, second_dict):
    for name in second_dict:
        if name in first_dict:
            raise ValueError(f"Found name collision. Found duplicate with name {name}")


class ComposeEvaluator(Evaluator):
    """
    Composes multiple evaluators.
    """

    def __init__(self, evaluators: Sequence[Evaluator]):
        self.evaluators = evaluators
        self.metric_infos = _aggregate_dictionaries([evaluator.get_metric_infos() for evaluator in self.evaluators])
        self.metrics = _aggregate_dictionaries([evaluator.get_metrics() for evaluator in self.evaluators])
        self.tracked_values = _aggregate_dictionaries([evaluator.get_tracked_values() for evaluator in self.evaluators])

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def evaluate(self):
        metric_values = {}
        for evaluator in self.evaluators:
            metric_values.update(evaluator.evaluate())

        return metric_values


class ComposeTrainEvaluator(TrainEvaluator):
    """
    Composes multiple train evaluators.
    """

    def __init__(self, evaluators: Sequence[TrainEvaluator]):
        self.evaluators = evaluators
        self.metric_infos = _aggregate_dictionaries([evaluator.get_metric_infos() for evaluator in self.evaluators])
        self.metrics = _aggregate_dictionaries([evaluator.get_metrics() for evaluator in self.evaluators])
        self.tracked_values = _aggregate_dictionaries([evaluator.get_tracked_values() for evaluator in self.evaluators])

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def evaluate_batch(self, output):
        metric_values = {}
        for evaluator in self.evaluators:
            metric_values.update(evaluator.evaluate_batch(output))

        return metric_values
