import json
from abc import ABC, abstractmethod


class ExperimentResult:
    """
    Result object of an Experiment. Contains a score for the fitted model and also additional summary metadata.
    """

    def __init__(self, score: float, score_name: str, score_epoch: int = -1, summary: dict = None):
        self.score = score
        self.score_name = score_name
        self.score_epoch = score_epoch
        self.summary = summary if summary is not None else {}

    def __str__(self):
        exp_result_str = f"Score name: {self.score_name}\nScore value: {self.score:.3f}\n"
        if self.score_epoch != -1:
            exp_result_str += f"Score epoch: {self.score_epoch}\n"
        exp_result_str += f"Summary: {json.dumps(self.summary, indent=2)}"
        return exp_result_str


class Experiment(ABC):
    """
    Abstract experiment class. Wraps a model and trainer to create an abstraction for experiment running.
    """

    @abstractmethod
    def run(self, config: dict, context: dict = None) -> ExperimentResult:
        """
        Runs the experiment with the given configuration. Usually fits a model and returns an ExperimentResult object with the score for the
        experiment/model, the larger the better, and additional summary metadata. An example for a score is returning the negative validation loss.
        :param config: configurations dictionary for the experiment
        :param context: optional context dictionary with additional information (e.g. can contain an ExperimentsPlan configuration)
        """
        raise NotImplementedError
