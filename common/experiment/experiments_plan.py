from __future__ import annotations

import itertools
import json
from typing import List


class ExperimentsPlan:
    """
    Experiment plan configuration object. Allows loading and parsing of an experiments plan JSON configuration.
    """

    ESCAPE_CHARACTERS = ['.', ',']
    REMOVE_CHARACTERS = ['[', ']', " "]

    @staticmethod
    def load_from(plan_config_path: str) -> ExperimentsPlan:
        with open(plan_config_path) as f:
            raw_plan_config = json.load(f)
            return ExperimentsPlan(raw_plan_config)

    def __init__(self, raw_plan_config: dict):
        self.raw_plan_config = raw_plan_config

        self.name = self.raw_plan_config["name"] if "name" in self.raw_plan_config else ""
        self.description = self.raw_plan_config["description"] if "description" in self.raw_plan_config else ""
        self.skip = self.raw_plan_config["skip"] if "skip" in self.raw_plan_config else 0
        self.repetitions = self.raw_plan_config["repetitions"] if "repetitions" in self.raw_plan_config else 1
        self.largest = self.raw_plan_config["largest"] if "largest" in self.raw_plan_config else True
        self.multiprocess = self.raw_plan_config["multiprocess"] if "multiprocess" in self.raw_plan_config else False
        self.num_parallel = self.raw_plan_config["num_parallel"] if "num_parallel" in self.raw_plan_config else 1
        self.gpu_ids_pool = self.raw_plan_config["gpu_ids_pool"] if "gpu_ids_pool" in self.raw_plan_config else []
        self.experiments_configurations_seq = self.__extract_experiments_configurations()

    def __extract_experiments_configurations(self) -> List[dict]:
        experiments_configurations_seq = []

        for configuration_def in self.raw_plan_config["configurations"]:
            base_config = configuration_def["base_config"]
            options = configuration_def["options"] if "options" in configuration_def else {}

            experiments_configurations = self.__create_experiment_configurations_for_base_config(base_config, options)
            experiments_configurations_seq.extend(experiments_configurations)

        return experiments_configurations_seq

    def __create_experiment_configurations_for_base_config(self, base_config: dict, options: dict) -> List[dict]:
        if len(options) == 0:
            config = base_config.copy()
            config = self.__format_experiment_config(config)
            return [config]

        field_names = options.keys()
        config_values = [options[field_name] for field_name in field_names]

        experiments_configurations = []
        all_options_iterator = itertools.product(*config_values)
        for values in all_options_iterator:
            config = base_config.copy()
            for field_name, config_value in zip(field_names, values):
                config[field_name] = config_value

            config = self.__format_experiment_config(config)
            experiments_configurations.append(config)

        return experiments_configurations

    def __format_experiment_config(self, config: dict):
        config = {k: config[k].format(**config) if type(config[k]) is str else config[k] for k in config}

        experiment_name = config.get("experiment_name")
        if not experiment_name:
            return config

        for ch in self.ESCAPE_CHARACTERS:
            experiment_name = experiment_name.replace(ch, "-")

        for ch in self.REMOVE_CHARACTERS:
            experiment_name = experiment_name.replace(ch, "")

        config["experiment_name"] = experiment_name

        return config
