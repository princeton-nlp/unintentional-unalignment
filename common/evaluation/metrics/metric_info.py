from typing import Dict, Sequence

from .metric import Metric, ScalarMetric


class MetricInfo:

    def __init__(self, name: str, metric: Metric, tag: str = "", save_epoch_values: bool = True, num_per_epoch_batch_histories_to_save: int = 0):
        """
        :param name: Name of the metric.
        :param metric: Metric object.
        :param tag: Optional tag for the metric. The tag can be used to aggregate metrics and plot all metrics with the
        same tag together.
        :param save_epoch_values: Flag whether or not to accumulate value history through epochs (if it is updated).
        :param num_per_epoch_batch_histories_to_save: Number of last epochs to save the per batch history for (if it is updated). -1 for saving all
        epoch batch histories.
        """
        self.name = name
        self.metric = metric
        self.tag = tag if tag != "" else name
        self.save_epoch_values = save_epoch_values
        self.num_per_epoch_batch_histories_to_save = num_per_epoch_batch_histories_to_save
        self.is_scalar = isinstance(metric, ScalarMetric)


def metric_info_seq_to_dict(metric_info_seq: Sequence[MetricInfo]) -> Dict[str, MetricInfo]:
    """
    :param metric_info_seq: Sequence of MetricInfo object.
    :return: Dict of MetricInfo where the key is the metric name. Will raise a ValueError exception if there are metrics with the same name.
    """
    __verify_no_duplicate_metric_names(metric_info_seq)
    return {metric_info.name: metric_info for metric_info in metric_info_seq}


def get_metric_dict_from_metric_info_seq(metric_info_seq: Sequence[MetricInfo]) -> Dict[str, Metric]:
    """
    :param metric_info_seq: Sequence of MetricInfo object.
    :return: Dict of Metric objects where the key is the metric name. Will raise a ValueError exception if there are metrics with the same name.
    """
    __verify_no_duplicate_metric_names(metric_info_seq)
    return {metric_info.name: metric_info.metric for metric_info in metric_info_seq}


def __verify_no_duplicate_metric_names(metric_info_seq: Sequence[MetricInfo]):
    """
    Raises a ValueError if there exists metric infos with duplicate names.
    :param metric_info_seq: Sequence of MetricInfo object.
    """
    existing_names = set()
    for metric_info in metric_info_seq:
        if metric_info.name in existing_names:
            raise ValueError(f"Found metrics with a duplicate name of '{metric_info.name}' in the same metric info sequence.")

        existing_names.add(metric_info.name)


def get_metric_dict_from_metric_info_dict(metric_info_dict: Dict[str, MetricInfo]) -> Dict[str, Metric]:
    """
    :param metric_info_dict: Dict of MetricInfo objects where the keys are the metric names.
    :return: Dict of Metric objects where the key is the metric name
    """
    return {metric_name: metric_info.metric for metric_name, metric_info in metric_info_dict.items()}
