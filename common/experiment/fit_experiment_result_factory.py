import bisect
from typing import Tuple, Dict, Union

import numpy as np
import torch.nn as nn

import common.utils.module as module_utils
from .experiment import ExperimentResult
from ..train.fit_output import FitOutput
from ..train.tracked_value import TrackedValue


class FitExperimentResultFactory:
    """
    Static factory for creating ExperimentResult objects from the returned FitOutput of a Trainer.
    """

    @staticmethod
    def create_from_best_metric_score(model: nn.Module, metric_name: str, fit_output: FitOutput, largest=True, is_train_metric: bool = False,
                                      additional_metadata: dict = None) -> ExperimentResult:
        """
        Creates an experiment result with the best score value of the given metric from all epochs.
        :param model: PyTorch model.
        :param metric_name: name of the metric that defines the score.
        :param fit_output: result of the Trainer fit method.
        :param largest: whether larger is better.
        :param is_train_metric: if true, then the score will be extracted from the train tracked values.
        :param additional_metadata: additional metadata summarizing experiment to populate the experiment result with.
        :return: ExperimentResult with the score name, best score value, best score epoch and additional metadata with the rest of the
        metric values from the epoch that achieved the best score.
        """
        tracked_values = fit_output.train_tracked_values if is_train_metric else fit_output.val_tracked_values
        relevant_tracked_value = tracked_values[metric_name]

        best_score, best_score_epoch = FitExperimentResultFactory.__get_best_score_and_epoch_of_tracked_value(relevant_tracked_value, largest=largest)

        experiment_result_summary = FitExperimentResultFactory.__create_summary(model, fit_output, metric_name, largest, is_train_metric,
                                                                                best_score_epoch)
        if additional_metadata:
            experiment_result_summary.update(additional_metadata)

        return ExperimentResult(best_score, metric_name, best_score_epoch, experiment_result_summary)

    @staticmethod
    def create_from_last_metric_score(model: nn.Module, metric_name: str, fit_output: FitOutput, largest=True, is_train_metric: bool = False,
                                      additional_metadata: dict = None) -> ExperimentResult:
        """
        Creates an experiment result with the last score value of the given metric from the last training epoch.
        :param model: PyTorch model.
        :param metric_name: name of the metric that defines the score.
        :param fit_output: result of the Trainer fit method.
        :param largest: whether larger is better.
        :param is_train_metric: if true, then the score will be extracted from the train tracked values.
        :param additional_metadata: additional metadata summarizing experiment to populate the experiment result with.
        :return: ExperimentResult with the score name, last score value, last score epoch or -1 if no epoch history is saved,
        and additional metadata with the rest of the metric values from the last epoch.
        """
        tracked_values = fit_output.train_tracked_values if is_train_metric else fit_output.val_tracked_values

        relevant_tracked_value = tracked_values[metric_name]
        score = relevant_tracked_value.current_value
        if score is None:
            score = -np.inf if largest else np.inf

        score_epoch = relevant_tracked_value.epoch_last_updated

        experiment_result_summary = FitExperimentResultFactory.__create_summary(model, fit_output, metric_name, largest, is_train_metric)
        if additional_metadata:
            experiment_result_summary.update(additional_metadata)

        return ExperimentResult(score, metric_name, score_epoch, experiment_result_summary)

    @staticmethod
    def create_from_best_metric_with_prefix_score(model: nn.Module, metric_name_prefix: str, fit_output: FitOutput, largest=True,
                                                  is_train_metric: bool = False, summary: dict = None) -> ExperimentResult:
        """
        Creates an experiment result with the best score value of the out of all the metrics that start with the given prefix
        from all of the training epochs.
        :param model: PyTorch model.
        :param metric_name_prefix: prefix of the metric names that are considered when getting the best score.
        :param fit_output: result of the Trainer fit method.
        :param largest: whether larger is better.
        :param is_train_metric: if true, then the score will be extracted from the train tracked values.
        :param summary: additional metadata summarizing experiment to populate the experiment result with.
        :return: ExperimentResult with the score name, best score value, best score epoch and additional metadata with the rest of the
        metric values from the epoch that achieved the best score.
        """
        tracked_values = fit_output.train_tracked_values if is_train_metric else fit_output.val_tracked_values
        best_score, metric_name, best_score_epoch = FitExperimentResultFactory.__get_best_metric_score_and_name_with_prefix(metric_name_prefix,
                                                                                                                            tracked_values,
                                                                                                                            largest=largest)

        experiment_result_summary = FitExperimentResultFactory.__create_summary(model, fit_output, metric_name, largest, is_train_metric,
                                                                                best_score_epoch)
        if summary:
            experiment_result_summary.update(summary)

        return ExperimentResult(best_score, metric_name, best_score_epoch, experiment_result_summary)

    @staticmethod
    def create_from_last_metric_with_prefix_score(model: nn.Module, metric_name_prefix: str, fit_output: FitOutput, largest=True,
                                                  is_train_metric: bool = False, additional_metadata: dict = None) -> ExperimentResult:
        """
        Creates an experiment result with the best last score value of the of the metrics that start with the given prefix.
        :param model: PyTorch model.
        :param metric_name_prefix: prefix of the metric names that are considered when getting the best score.
        :param fit_output: result of the Trainer fit method.
        :param largest: whether larger is better.
        :param is_train_metric: if true, then the score will be extracted from the train tracked values.
        :param additional_metadata: additional metadata summarizing experiment to populate the experiment result with.
        :return: ExperimentResult with the score name, last score value, last score epoch or -1 if no epoch history is saved,
        and additional metadata with the rest of the metric values from the last epoch.
        """
        tracked_values = fit_output.train_tracked_values if is_train_metric else fit_output.val_tracked_values
        score, metric_name, score_epoch = FitExperimentResultFactory.__get_best_last_metric_score_and_name_with_prefix(metric_name_prefix,
                                                                                                                       tracked_values,
                                                                                                                       largest=largest)

        experiment_result_summary = FitExperimentResultFactory.__create_summary(model, fit_output, metric_name, largest, is_train_metric)
        if additional_metadata:
            experiment_result_summary.update(additional_metadata)

        return ExperimentResult(score, metric_name, score_epoch, experiment_result_summary)

    @staticmethod
    def __create_summary(model: nn.Module, fit_output: FitOutput, metric_name: str,
                         largest: bool, is_train_metric: bool, best_score_epoch: int = None) -> dict:
        train_tracked_values = fit_output.train_tracked_values
        val_tracked_values = fit_output.val_tracked_values
        other_tracked_values = fit_output.value_store.tracked_values

        summary = {
            "num_model_parameters": module_utils.get_number_of_parameters(model),
            "last_epoch": fit_output.last_epoch,
            "score_name": metric_name,
            "score_largest": largest,
            "is_train_metric": is_train_metric
        }

        last_score_epoch_tracked_values = {}

        FitExperimentResultFactory.__populate_last_score_epoch_tracked_values(last_score_epoch_tracked_values, train_tracked_values)
        FitExperimentResultFactory.__populate_last_score_epoch_tracked_values(last_score_epoch_tracked_values, val_tracked_values)
        FitExperimentResultFactory.__populate_last_score_epoch_tracked_values(last_score_epoch_tracked_values, other_tracked_values)

        summary["last_tracked_values"] = last_score_epoch_tracked_values

        if best_score_epoch is not None:
            best_score_epoch_tracked_values = {}

            FitExperimentResultFactory.__populate_best_score_tracked_values(best_score_epoch_tracked_values, best_score_epoch, train_tracked_values)
            FitExperimentResultFactory.__populate_best_score_tracked_values(best_score_epoch_tracked_values, best_score_epoch, val_tracked_values)
            FitExperimentResultFactory.__populate_best_score_tracked_values(best_score_epoch_tracked_values, best_score_epoch, other_tracked_values)

            summary["best_score_epoch_tracked_values"] = best_score_epoch_tracked_values
            summary["best_score_epoch"] = best_score_epoch

        if fit_output.exception_occured():
            summary["exception"] = str(fit_output.exception)

        return summary

    @staticmethod
    def __populate_last_score_epoch_tracked_values(last_score_epoch_tracked_values: dict, tracked_values: Dict[str, TrackedValue]):
        for name, tracked_value in tracked_values.items():
            last_score_epoch_tracked_values[name] = {
                "value": tracked_value.current_value,
                "epoch": tracked_value.epoch_last_updated
            }

    @staticmethod
    def __populate_best_score_tracked_values(best_score_epoch_tracked_values: dict, score_epoch: int, tracked_values: Dict[str, TrackedValue]):
        for name, tracked_value in tracked_values.items():
            value, epoch = FitExperimentResultFactory.__get_last_value_up_to_epoch(tracked_value, score_epoch)
            best_score_epoch_tracked_values[name] = {"value": value, "epoch": score_epoch}

    @staticmethod
    def __get_last_value_up_to_epoch(tracked_value: TrackedValue, epoch: int) -> Tuple[Union[float, None], int]:
        epoch_index = bisect.bisect_right(tracked_value.epochs_with_values, epoch) - 1
        if epoch_index < 0:
            return None, -1

        return tracked_value.epoch_values[epoch_index], tracked_value.epochs_with_values[epoch_index]

    @staticmethod
    def __get_best_metric_score_and_name_with_prefix(metric_name_prefix: str,
                                                     tracked_values: Dict[str, TrackedValue],
                                                     largest: bool = True) -> Tuple[float, str, int]:
        best_scores = []
        names = []
        best_score_epochs = []
        for name, tracked_value in tracked_values.items():
            if name.startswith(metric_name_prefix):
                best_score, best_score_epoch = FitExperimentResultFactory.__get_best_score_and_epoch_of_tracked_value(tracked_value, largest=largest)

                best_scores.append(best_score)
                names.append(name)
                best_score_epochs.append(best_score_epoch)

        index_of_max = np.argmax(best_scores)
        return best_scores[index_of_max], names[index_of_max], best_score_epochs[index_of_max]

    @staticmethod
    def __get_best_last_metric_score_and_name_with_prefix(metric_name_prefix: str,
                                                          tracked_values: Dict[str, TrackedValue],
                                                          largest: bool = True) -> Tuple[float, str, int]:
        scores = []
        names = []
        score_epochs = []
        for name, tracked_value in tracked_values.items():
            if name.startswith(metric_name_prefix):
                score = tracked_value.current_value
                if score is None:
                    score = -np.inf if largest else np.inf

                score_epoch = tracked_value.epoch_last_updated

                scores.append(score)
                names.append(name)
                score_epochs.append(score_epoch)

        index_of_max = np.argmax(scores)
        return scores[index_of_max], names[index_of_max], score_epochs[index_of_max]

    @staticmethod
    def __get_best_score_and_epoch_of_tracked_value(tracked_value, largest: bool = True):
        if len(tracked_value.epoch_values) == 0:
            if tracked_value.current_value is None:
                best_score = -np.inf if largest else np.inf
                best_score_epoch = -1
            else:
                best_score = tracked_value.current_value
                best_score_epoch = tracked_value.epoch_last_updated

            return best_score, best_score_epoch

        get_best_score_index_fn = np.argmax if largest else np.argmin

        index_of_best_score = get_best_score_index_fn(tracked_value.epoch_values)
        best_score = tracked_value.epoch_values[index_of_best_score]
        best_score_epoch = tracked_value.epochs_with_values[index_of_best_score]
        return best_score, best_score_epoch
