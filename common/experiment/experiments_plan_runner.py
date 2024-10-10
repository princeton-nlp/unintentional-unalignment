import json
import os
from datetime import datetime

from .experiment import Experiment
from .experiments_plan import ExperimentsPlan
from ..train.tuning import Tuner
from ..utils import logging as logging_utils


class ExperimentsPlanRunner:
    """
    Runs a given experiment (possibly multiple times) according to a configuration file.
    """

    @staticmethod
    def add_experiments_plan_runner_specific_args(parser):
        parser.add_argument("--plan_config_path", type=str, required=True, help="path to the plan config file")
        parser.add_argument("--disable_console_log", action='store_true', help="do not log experiments runner logs to console")
        parser.add_argument("--save_logs", action='store_true', help="save logs to file")
        parser.add_argument("--log_dir", type=str, default="", help="directory to save experiments runner log file in (default is cwd)")
        parser.add_argument("--log_file_name_prefix", type=str, default="plan", help="prefix for the log file name")

    def run(self, plan_config_path: str, experiment: Experiment, disable_console_log: bool = False, save_logs: bool = False,
            log_dir: str = "", log_file_name_prefix: str = ""):
        """
        Runs the experiment (possibly multiple times) with configurations as defined in the given configuration file.
        :param plan_config_path: path to a configuration file defining the experiments plan
        :param experiment: Experiment object that will be run according to configurations
        :param disable_console_log: do not log experiments runner logs to console
        :param save_logs: save logs to file
        :param log_dir: directory to save experiments runner log file in, default is current working directory
        :param log_file_name_prefix: prefix for the log file name
        """
        experiments_plan = ExperimentsPlan.load_from(plan_config_path)
        configurations_seq = experiments_plan.experiments_configurations_seq
        log_dir = log_dir if log_dir else os.getcwd()

        logger = logging_utils.create_logger(console_logging=not disable_console_log, file_logging=save_logs, log_dir=log_dir,
                                             log_file_name_prefix=log_file_name_prefix)

        start_time = datetime.utcnow()
        logger.info(f"Starting experiments plan execution\n"
                    f"Name: {experiments_plan.name}\n"
                    f"Description: {experiments_plan.description}\n"
                    f"Number of experiments: {len(configurations_seq)}\n"
                    f"Plan configuration:\n{json.dumps(experiments_plan.raw_plan_config, indent=2)}")

        context = {
            "experiments_plan_config": experiments_plan.raw_plan_config
        }

        tuner = Tuner(experiment, context=context, largest=experiments_plan.largest,
                      multiprocess=experiments_plan.multiprocess,
                      num_parallel=experiments_plan.num_parallel,
                      gpu_ids_pool=experiments_plan.gpu_ids_pool,
                      logger=logger)
        tuner.preset_options_search(configurations_seq, skip=experiments_plan.skip, repetitions=experiments_plan.repetitions)

        time_took = datetime.utcnow() - start_time
        logger.info(f"Finished experiments plan execution. Time took: {time_took}")
