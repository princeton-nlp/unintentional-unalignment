import itertools
import json
import logging
import threading
import traceback
from collections import deque
from typing import List

import numpy as np
import torch.cuda
import torch.multiprocessing as multiprocessing

from ...experiment import ExperimentResult, Experiment


class TuneResult:
    """
    Result of a hyperparameter tuning process.
    """

    def __init__(self, largest: bool = True):
        """
        :param largest: whether larger score is better
        """
        self.largest = largest
        self.config_results = []
        self.best_config_result = ConfigResult({}, largest=largest)

    def __is_better(self, score1, score2):
        return score1 > score2 if self.largest else score1 < score2

    def add_config_result(self, config_result):
        self.config_results.append(config_result)
        if self.__is_better(config_result.get_score(), self.best_config_result.get_score()):
            self.best_config_result = config_result


class ConfigResult:
    """
    Result for a certain hyperparameter configuration. Wraps multiple ExperimentResults.
    """

    def __init__(self, config: dict, score_reduction: str = "mean", largest: bool = True):
        """
        :param config: configuration dictionary.
        :param score_reduction: determines score reduction method. Supports: 'mean', 'media', 'max', 'min'/
        :param largest: whether larger score is better.
        """
        self.config = config
        self.score_reduction = score_reduction.lower()
        self.score_reduce_fn = self.__get_score_reduce_fn(self.score_reduction)
        self.largest = largest
        self.worst_score = -np.inf if largest else np.inf

        self.experiment_results = []
        self.best_experiment_result = ExperimentResult(self.worst_score, "")

    def __is_better(self, score1, score2):
        return score1 > score2 if self.largest else score1 < score2

    @staticmethod
    def __get_score_reduce_fn(score_reduction: str):
        if score_reduction == "mean":
            return np.mean
        elif score_reduction == "median":
            return np.median
        elif score_reduction == "max":
            return np.max
        elif score_reduction == "min":
            return np.min

        raise ValueError(f"Unsupported score reduction type: {score_reduction}. Supported types are: 'mean', 'median', 'max', 'min'.")

    def add_experiment_result(self, experiment_result: ExperimentResult):
        self.experiment_results.append(experiment_result)
        if self.__is_better(experiment_result.score, self.best_experiment_result.score):
            self.best_experiment_result = experiment_result

    def get_score(self):
        return self.score_reduce_fn([experiment_result.score for experiment_result in self.experiment_results]) \
            if len(self.experiment_results) != 0 else self.worst_score

    def get_score_std(self):
        return np.std([experiment_result.score for experiment_result in self.experiment_results]) if len(self.experiment_results) != 0 else 0


class WorkerArguments:
    """
    Arguments for multiprocess workers.
    """

    def __init__(self, experiment: Experiment, config: dict, context: dict, config_index: int, num_configs: int, manage_gpu: bool,
                 largest: bool = True, repetitions: int = 1, score_reduction: str = "mean"):
        self.experiment = experiment
        self.config = config
        self.context = context
        self.config_index = config_index
        self.num_configs = num_configs
        self.manage_gpu = manage_gpu
        self.largest = largest
        self.repetitions = repetitions
        self.score_reduction = score_reduction


class Tuner:
    """
    Tunes hyperparameters for a given experiment.
    """

    GPU_IDS_CONFIG_FIELD_NAME = "gpu_ids"

    def __init__(self, experiment: Experiment, context: dict = None, largest: bool = True,
                 multiprocess: bool = False, num_parallel: int = 1, gpu_ids_pool: List[int] = None,
                 logger: logging.Logger = None):
        """
        :param experiment: Experiment to run
        :param context: optional context dictionary of the experiment (e.g. can contain an ExperimentsPlan configuration)
        :param largest: whether for the score returned in the ExperimentResult larger is better.
        :param multiprocess: run experiments in different processes.
        :param num_parallel: number of processes to run in parallel at a time.
        :param gpu_ids_pool: pool of gpu ids. If not None, will manage GPUs allocation for the experiments. Each experiment is allocated an
        available GPU from the pool by setting a 'gpu_ids' parameter in its configuration. Must contain at least 'num_parallel' gpu ids
        if 'multiprocess' is set to True.
        :param logger: logger to use for logging tuning related logs.
        """
        self.experiment = experiment
        self.context = context
        self.largest = largest
        self.largest_log_str = "maximal" if largest else "minimal"
        self.multiprocess = multiprocess
        self.num_parallel = num_parallel
        self.gpu_ids_pool = gpu_ids_pool if gpu_ids_pool is not None else []
        self.manage_gpu = len(self.gpu_ids_pool) > 0

        if self.gpu_ids_pool and self.multiprocess and len(self.gpu_ids_pool) < self.num_parallel:
            raise ValueError(f"GPUs pool must contain at least the number of parallel processes ids. Only {len(self.gpu_ids_pool)} GPU ids given "
                             f"while 'num_parallel' is {self.num_parallel}")

        if self.multiprocess and self.num_parallel < 1:
            raise ValueError(f"Number of parallel processes must be at least 1. Given 'num_parallel' is {self.num_parallel}")

        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def __create_finished_search_log_message(self, search_type_name: str, tune_result: TuneResult) -> str:
        return f"Finished {search_type_name} search.\n" \
               f"Best config:\n{json.dumps(tune_result.best_config_result.config, indent=2)}\n" \
               f"Best ({self.largest_log_str}) config score value: {tune_result.best_config_result.get_score()}\n" \
               f"Best config score std: {tune_result.best_config_result.get_score_std()}\n" \
               f"Best config best experiment result:\n{tune_result.best_config_result.best_experiment_result}"

    def random_search(self, base_config, config_samplers, n_iter=10, repetitions=1, score_reduction="mean") -> TuneResult:
        """
        Runs random search hyperparameter tuning.
        :param base_config: base dictionary of configurations for the experiments.
        :param config_samplers: dictionary of configuration name to a callable that samples a value.
        :param n_iter: number of configuration settings that are sampled.
        :param repetitions: number of times to repeat the experiment per configuration.
        :param score_reduction: score reduction method per configuration.
        :return: TuneResult object with the results of the tuning process.
        """
        self.logger.info(f"Starting random search for {n_iter} iterations.\n"
                         f"Base config:\n{json.dumps(base_config, indent=2)}\n"
                         f"Config with samplers: {config_samplers.keys()}")
        config_seq = []
        for i in range(n_iter):
            config = base_config.copy()
            for config_name, sampler in config_samplers.items():
                config[config_name] = sampler()

            config_seq.append(config)

        tune_result = self.__run_search(config_seq, skip=0, repetitions=repetitions, score_reduction=score_reduction)
        self.logger.info(self.__create_finished_search_log_message(search_type_name="random", tune_result=tune_result))
        return tune_result

    def grid_search(self, base_config, config_options, skip=0, repetitions=1, score_reduction="mean") -> TuneResult:
        """
        Runs grid search hyperparameter tuning.
        :param base_config: base configuration dictionary for the experiments.
        :param config_options: dictionary of configuration name to a sequence of values.
        :param skip: number of configurations from the start to skip.
        :param repetitions: number of times to repeat the experiment per parameter configuration.
        :param score_reduction: score reduction method per configuration.
        :return: TuneResult object with the results of the tuning process.
        """
        config_names = config_options.keys()
        config_values = [config_options[config_name] for config_name in config_names]
        num_configs = np.prod([len(values) for values in config_values])

        self.logger.info(f"Starting grid search for {num_configs} configurations.\n"
                         f"Base config:\n{json.dumps(base_config, indent=2)}\n"
                         f"Config options: {json.dumps(config_options, indent=2)}")
        config_seq = []
        all_options_iterator = itertools.product(*config_values)
        for i, values in enumerate(all_options_iterator):
            config = base_config.copy()
            for name, value in zip(config_names, values):
                config[name] = value

            config_seq.append(config)

        tune_result = self.__run_search(config_seq, skip=skip, repetitions=repetitions, score_reduction=score_reduction)
        self.logger.info(self.__create_finished_search_log_message(search_type_name="grid", tune_result=tune_result))
        return tune_result

    def preset_options_search(self, configs_seq, skip=0, repetitions=1, score_reduction="mean") -> TuneResult:
        """
        Runs hyperparameter search for the given preset configuration.
        :param configs_seq: sequence of configuration dictionaries to try.
        :param skip: number of configurations from the start to skip.
        :param repetitions: number of times to repeat the experiment per configuration.
        :param score_reduction: score reduction method per configuration.
        :return: TuneResult object with the results of the tuning process.
        """
        self.logger.info(f"Starting preset options search for {len(configs_seq)} options.")
        tune_result = self.__run_search(configs_seq, skip=skip, repetitions=repetitions, score_reduction=score_reduction)
        self.logger.info(self.__create_finished_search_log_message(search_type_name="preset options", tune_result=tune_result))
        return tune_result

    def __run_search(self, configs_seq, skip: int = 0, repetitions: int = 1, score_reduction: str = "mean") -> TuneResult:
        """
        Runs hyperparameter search on the given configurations.
        :param configs_seq: sequence of configuration dictionaries.
        :param skip: number of configurations from the start to skip.
        :param repetitions: number of times to repeat the experiment per configuration.
        :param score_reduction: score reduction method per configuration.
        :return: TuneResult object with the results of the tuning process.
        """
        if self.multiprocess:
            return self.__multiprocess_run_search(configs_seq, skip=skip, repetitions=repetitions, score_reduction=score_reduction)

        return self.__run_search_in_current_process(configs_seq, skip=skip, repetitions=repetitions, score_reduction=score_reduction)

    def __run_search_in_current_process(self, configs_seq, skip: int = 0, repetitions: int = 1, score_reduction: str = "mean") -> TuneResult:
        """
        Runs hyperparameter search on the given configurations in the current process (no multiprocessing).
        :param configs_seq: sequence of configuration dictionaries.
        :param skip: number of configurations from the start to skip.
        :param repetitions: number of times to repeat the experiment per configuration.
        :param score_reduction: score reduction method per configuration.
        :return: TuneResult object with the results of the tuning process.
        """
        num_configs = len(configs_seq)
        tune_result = TuneResult(largest=self.largest)
        gpu_ids_queue = deque(self.gpu_ids_pool)

        for i, config in enumerate(itertools.islice(configs_seq, skip, None)):
            config = config.copy()

            if self.manage_gpu:
                gpu_id = gpu_ids_queue.pop()
                config[self.GPU_IDS_CONFIG_FIELD_NAME] = [gpu_id]

            config_result = ConfigResult(config, score_reduction=score_reduction, largest=self.largest)

            self.logger.info(_create_start_config_experiment_log_message(skip + i, num_configs, repetitions, config))
            for r in range(repetitions):
                self.logger.info(_create_start_repetition_log_message(skip + i, num_configs, r, repetitions))

                experiment_result = self.experiment.run(config, context=self.context)
                config_result.add_experiment_result(experiment_result)

                self.logger.info(_create_finish_repetition_log_message(skip + i, num_configs, r, repetitions, experiment_result))

            self.logger.info(_create_finish_config_experiment_log_message(skip + i, num_configs, repetitions, self.largest_log_str, config_result))
            tune_result.add_config_result(config_result)

            if self.manage_gpu:
                gpu_ids_queue.appendleft(gpu_id)

        return tune_result

    def __multiprocess_run_search(self, configs_seq, skip: int = 0, repetitions: int = 1, score_reduction: str = "mean") -> TuneResult:
        """
        Runs hyperparameter search on the given configurations using multiple processes.
        :param configs_seq: sequence of configuration dictionaries.
        :param skip: number of iterations from the start to skip.
        :param repetitions: number of times to repeat the experiment per configuration.
        :param score_reduction: score reduction method per configuration.
        :return: TuneResult object with the results of the tuning process.
        """
        # Necessary for PyTorch CUDA GPU training with multiple processes. See https://pytorch.org/docs/stable/notes/multiprocessing.html
        multiprocessing.set_start_method("spawn")

        tune_result = TuneResult(largest=self.largest)

        gpu_ids_queue = multiprocessing.Queue()
        for gpu_id in self.gpu_ids_pool:
            gpu_ids_queue.put(gpu_id)

        log_queue = multiprocessing.Queue()
        logger_thread = threading.Thread(target=_logger_thread, args=(log_queue, self.logger))
        logger_thread.start()

        workers_arguments_list = self.__create_workers_arguments_list(configs_seq, skip, repetitions, score_reduction)
        with multiprocessing.Pool(processes=self.num_parallel, initializer=_exp_worker_init, initargs=[gpu_ids_queue, log_queue]) as pool:
            config_results = pool.map(_exp_worker, workers_arguments_list, chunksize=1)

            # closes logger thread by passing None message
            log_queue.put(None)
            logger_thread.join()

            for config_result in config_results:
                if config_result is not None:
                    tune_result.add_config_result(config_result)

            return tune_result

    def __create_workers_arguments_list(self, configs_seq, skip: int, repetitions: int = 1, score_reduction: str = "mean") -> List[WorkerArguments]:
        arguments_list = []
        for i, config in enumerate(itertools.islice(configs_seq, skip, None)):
            config_index = skip + i
            arguments_list.append(WorkerArguments(experiment=self.experiment,
                                                  config=config,
                                                  context=self.context,
                                                  config_index=config_index,
                                                  num_configs=len(configs_seq),
                                                  manage_gpu=self.manage_gpu,
                                                  largest=self.largest,
                                                  repetitions=repetitions,
                                                  score_reduction=score_reduction))

        return arguments_list


def _exp_worker(args: WorkerArguments) -> ConfigResult:
    config = args.config.copy()
    if "experiment_name" in config:
        config["experiment_name"] = config["experiment_name"] + f"_pid{multiprocessing.current_process().pid}"

    if args.manage_gpu:
        config[Tuner.GPU_IDS_CONFIG_FIELD_NAME] = [_exp_worker.gpu_ids_queue.get()]

    config_result = ConfigResult(config, score_reduction=args.score_reduction, largest=args.largest)
    try:
        _exp_worker.log_queue.put((logging.INFO,
                                   f"PID - {multiprocessing.current_process().pid} - "
                                   f"{_create_start_config_experiment_log_message(args.config_index, args.num_configs, args.repetitions, config)}"))

        for r in range(args.repetitions):
            _exp_worker.log_queue.put((logging.INFO,
                                       f"PID - {multiprocessing.current_process().pid} - "
                                       f"{_create_start_repetition_log_message(args.config_index, args.num_configs, r, args.repetitions)}"))

            experiment_result = args.experiment.run(config, context=args.context)
            config_result.add_experiment_result(experiment_result)

            _exp_worker.log_queue.put((logging.INFO,
                                       f"PID - {multiprocessing.current_process().pid} - "
                                       f"{_create_finish_repetition_log_message(args.config_index, args.num_configs, r, args.repetitions, experiment_result)}"))

        largest_log_str = "maximal" if args.largest else "minimal"
        fin_conf_exp_log = (logging.INFO,
                            f"PID - {multiprocessing.current_process().pid} - "
                            f"{_create_finish_config_experiment_log_message(args.config_index, args.num_configs, args.repetitions, largest_log_str, config_result)}")
        _exp_worker.log_queue.put(fin_conf_exp_log)

        return config_result
    except Exception:
        _exp_worker.log_queue.put((logging.ERROR,
                                   f"PID - {multiprocessing.current_process().pid} - "
                                   f"Exception in worker process {multiprocessing.current_process().pid}. {traceback.format_exc()}"))
    finally:
        if config[Tuner.GPU_IDS_CONFIG_FIELD_NAME]:
            torch.cuda.empty_cache()
        if args.manage_gpu:
            _exp_worker.gpu_ids_queue.put(config_result.config[Tuner.GPU_IDS_CONFIG_FIELD_NAME][0])


def _exp_worker_init(gpu_ids_queue: multiprocessing.Queue, log_queue: multiprocessing.Queue):
    _exp_worker.gpu_ids_queue = gpu_ids_queue
    _exp_worker.log_queue = log_queue


def _logger_thread(log_queue: multiprocessing.Queue, logger: logging.Logger):
    while True:
        log = log_queue.get()
        if log is None:
            break

        log_level, log_msg = log
        logger.log(log_level, log_msg)


def _create_start_config_experiment_log_message(config_index: int, num_configs: int, repetitions: int, config: dict) -> str:
    return f"Starting experiment for config {config_index + 1}/{num_configs} with {repetitions} repetitions:\n{json.dumps(config, indent=2)}"


def _create_finish_config_experiment_log_message(config_index: int, num_configs: int, repetitions: int, largest_log_str: str,
                                                 config_result: ConfigResult) -> str:
    return f"Finished experiment for config {config_index + 1}/{num_configs} with {repetitions} repetitions\n" \
           f"Config score value: {config_result.get_score()}\n" \
           f"Config score std: {config_result.get_score_std()}\n" \
           f"Config experiment result with best ({largest_log_str}) score:\n{config_result.best_experiment_result}"


def _create_start_repetition_log_message(config_index: int, num_configs: int, repetition: int, repetitions: int) -> str:
    return f"Starting repetition {repetition + 1}/{repetitions} for experiment {config_index + 1}/{num_configs}."


def _create_finish_repetition_log_message(config_index: int, num_configs: int, repetition: int, repetitions: int,
                                          experiment_result: ExperimentResult) -> str:
    return f"Finished repetition {repetition + 1}/{repetitions} for experiment {config_index + 1}/{num_configs}:\n{experiment_result}"
