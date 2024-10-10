import logging
import os
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from common.data.modules import DataModule
from common.evaluation.evaluators import TrainEvaluator, Evaluator, VoidEvaluator, TrainBatchOutputEvaluator
from common.experiment import FitExperimentBase, ExperimentResult
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from persona_experiments.data.persona_single_output_token_datamodule_accelerate import PersonaSingleOutputTokenDataModuleAccelerate
from persona_experiments.train.single_output_preference_based_trainer_accelerate import SingleOutputPreferenceBasedTrainerAccelerate
from persona_experiments.train.token_logits_and_probs_tracker_callback import TokenLogitsAndProbsTrackerCallback


class PersonaSingleExampleAccelerateExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser):
        FitExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--dataset", type=str, default="data_files/persona/ends-justify-means.jsonl", help="Dataset to use")
        parser.add_argument("--num_train_samples", type=int, default=1, help="Number of train samples to use (if < 0 will use the whole train set).")
        parser.add_argument("--train_samples_random_seed", type=int, default=-1, help="Random seed to use for selecting train samples")
        parser.add_argument("--answer_matching_behavior_to_use", type=str, default="No", help="If not empty, will use only training samples with the "
                                                                                              "given 'answer_matching_behavior' value. Should be "
                                                                                              "either 'Yes' or 'No'.")
        parser.add_argument("--batch_size", type=int, default=-1, help="train batch size")
        parser.add_argument("--output_tokens_matching_yes", nargs="+", type=str, default=["No"], help="Tokens corresponding to original answer 'Yes'")
        parser.add_argument("--output_tokens_matching_no", nargs="+", type=str, default=["Yes"], help="Tokens corresponding to original answer 'No'")

        parser.add_argument("--model", type=str, default="allenai/OLMo-1B-hf", help="Model to use")
        parser.add_argument("--model_cache_dir", type=str, default=None, help="Cache dir for downloaded Huggingface models")
        parser.add_argument("--load_model_checkpoint", type=str, default=None, help="If given, will load model from this checkpoint")
        parser.add_argument("--is_lora_checkpoint", action="store_true", help="Whether the checkpoint to load is a LoRA Peft checkpoint")
        parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA")
        parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
        parser.add_argument("--kl_coeff", type=float, default=0.1, help="KL divergence coefficient for DPO/IPO")
        parser.add_argument("--objective", type=str, default="dpo", help="Objective type to use. Supports 'dpo', 'ipo', and 'cross_entropy'.")

        parser.add_argument("--optimizer", type=str, default="rmsprop", help="Optimizer to use. Supports 'adam', 'rmsprop', and 'sgd'.")
        parser.add_argument("--lr", type=float, default=1e-7, help="learning rate")

        parser.add_argument("--log_top_token_logit_change_interval", type=int, default=10,
                            help="Log top token logit and prob changes every this number of epochs. Only relevant for full-batch optimiztion")
        parser.add_argument("--save_model", action="store_true", help="Save the model at the end of the experiment")
        parser.add_argument("--save_finegrained_token_metrics", action="store_true",
                            help="Track and save finegrained metrics for token logit and probability changes and token unembedding metrics")

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True, cache_dir=config["model_cache_dir"])

        if not config["is_lora_checkpoint"] or not config["load_model_checkpoint"]:
            model_path = config["load_model_checkpoint"] if config["load_model_checkpoint"] else config["model"]
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=config["model_cache_dir"]
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=config["model"],
                device_map="auto",
                trust_remote_code=True,
                cache_dir=config["model_cache_dir"]
            )
            model = PeftModel.from_pretrained(model=model, model_id=config["load_model_checkpoint"])
            model = model.merge_and_unload()

        if config["use_lora"]:
            # Using lora_alpha = 2 * lora_rank is according to Llama 3 8B recipe for LoRA finetuning:
            # https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora_single_device.yaml
            lora_config = LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_rank"] * 2,
                                     bias="all", target_modules="all-linear", use_rslora=True)
            model = get_peft_model(model, lora_config)

        accelerator = Accelerator()
        state["accelerator"] = accelerator
        state["device"] = None
        state["tokenizer"] = tokenizer
        state["model"] = accelerator.prepare(model)

        datamodule = PersonaSingleOutputTokenDataModuleAccelerate(path=config["dataset"], model=model, tokenizer=tokenizer,
                                                                  output_tokens_matching_yes=config["output_tokens_matching_yes"],
                                                                  output_tokens_matching_no=config["output_tokens_matching_no"],
                                                                  num_train_samples=config["num_train_samples"],
                                                                  accelerator=accelerator,
                                                                  answer_matching_behavior_to_use=config["answer_matching_behavior_to_use"],
                                                                  batch_size=config["batch_size"],
                                                                  random_seed=config["train_samples_random_seed"])
        datamodule.setup()
        if config["save_finegrained_token_metrics"]:
            state["initial_token_unembedding_metrics"] = self.__compute_token_unembedding_metrics(model, datamodule, accelerator,
                                                                                                  metric_name_prefix="initial_")
        return datamodule

    @torch.no_grad()
    def __compute_token_unembedding_metrics(self, model: nn.Module, datamodule: PersonaSingleOutputTokenDataModuleAccelerate,
                                            accelerator: Accelerator,
                                            metric_name_prefix: str = ""):
        train_batch = next(iter(datamodule.train_dataloader()))
        preferred_output_ids = train_batch["preferred_output_ids"].to(accelerator.device)
        dispreferred_output_ids = train_batch["dispreferred_output_ids"].to(accelerator.device)
        unembedding_weights = model.get_output_embeddings().weight.data.detach().to(accelerator.device)

        preferred_token_embeddings = unembedding_weights[preferred_output_ids]
        dispreferred_token_embeddings = unembedding_weights[dispreferred_output_ids]

        preferred_token_norms = preferred_token_embeddings.norm(dim=1)
        dispreferred_token_norms = dispreferred_token_embeddings.norm(dim=1)
        inner_products = (preferred_token_embeddings * dispreferred_token_embeddings).sum(dim=1)
        dispreferred_projection_coeffs = inner_products / preferred_token_norms
        preferred_norm_dispreferred_projection_coeff_diffs = preferred_token_norms - dispreferred_projection_coeffs
        norm_orthogonal_component_of_dispreferred_tokens = (dispreferred_token_embeddings
                                                            - (dispreferred_projection_coeffs / preferred_token_norms).view(-1, 1)
                                                            * preferred_token_embeddings).norm(dim=1)

        return {
            f"{metric_name_prefix}preferred_token_norms": preferred_token_norms.cpu(),
            f"{metric_name_prefix}dispreferred_token_norms": dispreferred_token_norms.cpu(),
            f"{metric_name_prefix}preferred_and_dispreferred_inner_products": inner_products.cpu(),
            f"{metric_name_prefix}dispreferred_projection_coeffs": dispreferred_projection_coeffs.cpu(),
            f"{metric_name_prefix}preferred_norm_dispreferred_projection_coeff_diffs": preferred_norm_dispreferred_projection_coeff_diffs.cpu(),
            f"{metric_name_prefix}norm_orthogonal_component_of_dispreferred_tokens": norm_orthogonal_component_of_dispreferred_tokens.cpu()
        }

    def create_model(self, datamodule: DataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        return state["model"]

    def create_train_and_validation_evaluators(self, model: nn.Module, datamodule: DataModule, device, config: dict, state: dict,
                                               logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        batch_output_metrics_to_track = ["train loss", "preferred logit", "preferred prob", "dispreferred logit", "dispreferred prob",
                                         "preferred logprob change", "dispreferred logprob change"]
        batch_output_metric_tags = ["train loss", "logits", "probs", "logits", "probs", "logprob change", "logprob change"]

        train_evaluator = TrainBatchOutputEvaluator(metric_names=batch_output_metrics_to_track, metric_tags=batch_output_metric_tags)
        return train_evaluator, VoidEvaluator()

    def create_additional_metadata_to_log(self, model: torch.nn.Module, datamodule: DataModule,
                                          config: dict, state: dict, logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        additional_metadata["num train samples"] = len(datamodule.dataset)
        first_samples = datamodule.dataset.select(torch.arange(min(len(datamodule.dataset), 5)))
        additional_metadata["data"] = {
            "inputs": [example["input"] for example in first_samples],
            "answer_matching_behavior": [example["answer_matching_behavior"] for example in first_samples],
            "preferred_output": [example["preferred_output"] for example in first_samples],
            "dispreferred_output": [example["dispreferred_output"] for example in first_samples]
        }

        return additional_metadata

    def customize_callbacks(self, callbacks_dict: OrderedDict, model: nn.Module, datamodule: DataModule, config: dict, state: dict,
                            logger: logging.Logger):
        if datamodule.batch_size <= 0 or datamodule.batch_size >= len(datamodule.dataset):
            callbacks_dict["token_tracker"] = TokenLogitsAndProbsTrackerCallback(tokenizer=state["tokenizer"],
                                                                                 logger=logger,
                                                                                 top_tokens_to_log=10,
                                                                                 epoch_log_interval=config["log_top_token_logit_change_interval"],
                                                                                 track_finegrained_token_metrics=config[
                                                                                     "save_finegrained_token_metrics"])

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="train loss", is_train_metric=True, largest=False, return_best_score=False)

    def create_trainer(self, model: nn.Module, datamodule: DataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger):
        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        elif config["optimizer"] == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
        else:
            raise ValueError(f"Optimizer '{config['optimizer']}' is not supported.")

        optimizer = state["accelerator"].prepare(optimizer)
        return SingleOutputPreferenceBasedTrainerAccelerate(model, tokenizer=state["tokenizer"], optimizer=optimizer,
                                                            accelerator=state["accelerator"], kl_coeff=config["kl_coeff"],
                                                            objective=config["objective"], train_evaluator=train_evaluator,
                                                            val_evaluator=val_evaluator, callback=callback,
                                                            track_logits_for_tokens=config["track_logits_for_tokens"],
                                                            gradient_accumulation=config[
                                                                "gradient_accumulation"] if "gradient_accumulation" in config else -1)

    def on_experiment_end(self, model: nn.Module, datamodule: PersonaSingleOutputTokenDataModuleAccelerate, trainer: Trainer, fit_output: FitOutput,
                          experiment_result: ExperimentResult, config: dict, state: dict, logger: logging.Logger):
        super().on_experiment_end(model, datamodule, trainer, fit_output, experiment_result, config, state, logger)

        if config["save_model"]:
            experiment_dir = state["experiment_dir"]
            model_dir_path = os.path.join(experiment_dir, f"model_epoch_{trainer.epoch}")
            model.save_pretrained(model_dir_path)

        if "token_tracker" in trainer.callback.callbacks:
            self.__update_experiment_result_with_token_logits_and_prob_change_metrics(experiment_result,
                                                                                      trainer.callback.callbacks["token_tracker"])

            if config["save_finegrained_token_metrics"]:
                initial_token_unembedding_metrics = state["initial_token_unembedding_metrics"]
                final_token_unembedding_metrics = self.__compute_token_unembedding_metrics(model, datamodule, state["accelerator"],
                                                                                           metric_name_prefix="final_")
                self.__save_finegrained_token_metrics(state["experiment_dir"], trainer.callback.callbacks["token_tracker"],
                                                      initial_token_unembedding_metrics, final_token_unembedding_metrics)

        if config["use_lora"]:
            trainable_params, all_param = model.get_nb_trainable_parameters()
            experiment_result.summary["trainable_params"] = trainable_params
            experiment_result.summary["all_params"] = all_param
            experiment_result.summary["trainable_params_percentage"] = 100 * trainable_params / all_param

        experiment_result.summary["num_train_samples"] = len(datamodule.dataset)
        experiment_result.summary["data"] = {
            "inputs": [example["input"] for example in datamodule.dataset],
            "answer_matching_behavior": [example["answer_matching_behavior"] for example in datamodule.dataset],
            "preferred_output": [example["preferred_output"] for example in datamodule.dataset],
            "dispreferred_output": [example["dispreferred_output"] for example in datamodule.dataset]
        }

        state["accelerator"].end_training()

    def __update_experiment_result_with_token_logits_and_prob_change_metrics(self, experiment_result: ExperimentResult,
                                                                             token_tracker_callback: TokenLogitsAndProbsTrackerCallback):
        experiment_result.summary["num_steps_preferred_prob_decreased"] = token_tracker_callback.num_steps_preferred_prob_decreased
        experiment_result.summary[
            "num_steps_train_loss_increase_when_preferred_prob_decreased"] = token_tracker_callback.num_steps_train_loss_increase_when_preferred_prob_decreased
        experiment_result.summary["min_pref_logprob_smaller_than_init"] = token_tracker_callback.min_pref_logprob_smaller_than_init
        experiment_result.summary["min_pref_logprob"] = token_tracker_callback.min_pref_logprob

    @torch.no_grad()
    def __save_finegrained_token_metrics(self, experiment_dir: str, tracker: TokenLogitsAndProbsTrackerCallback,
                                         initial_token_unembedding_metrics: dict, final_token_unembedding_metrics: dict,
                                         file_name: str = "token_metrics.pt"):
        token_metrics = {}
        token_metrics.update(initial_token_unembedding_metrics)
        token_metrics.update(final_token_unembedding_metrics)

        token_metrics["per_example_per_step_is_pref_logprob_smaller_than_init"] = torch.stack(
            tracker.per_step_per_example_is_pref_logprob_smaller_than_init, dim=1)
        token_metrics["per_example_per_step_did_preferred_prob_decrease"] = torch.stack(tracker.per_step_per_example_did_preferred_prob_decrease,
                                                                                        dim=1)
        token_metrics["per_step_did_train_loss_decrease"] = torch.tensor(tracker.per_step_did_train_loss_decrease)

        token_metrics["per_example_initial_preferred_token_logprob"] = tracker.initial_preferred_logprobs.cpu()
        token_metrics["per_example_initial_dispreferred_token_logprob"] = tracker.initial_dispreferred_logprobs.cpu()
        token_metrics["per_example_per_step_preferred_token_logprob_change"] = torch.stack(
            tracker.per_step_per_example_preferred_token_logprob_change,
            dim=1
        )
        token_metrics["per_example_per_step_dispreferred_token_logprob_change"] = torch.stack(
            tracker.per_step_per_example_dispreferred_token_logprob_change,
            dim=1
        )

        token_metrics["per_example_until_step_preferred_token_logprob_increase_rank"] = torch.stack(
            tracker.until_step_per_example_preferred_token_logprob_increase_rank, dim=1)
        token_metrics["per_example_until_step_preferred_token_prob_increase_rank"] = torch.stack(
            tracker.until_step_per_example_preferred_token_prob_increase_rank, dim=1)
        token_metrics["per_example_until_step_top_logprob_increase_token_ids"] = torch.stack(
            tracker.until_step_per_example_top_logprob_increase_token_ids, dim=1)
        token_metrics["per_example_until_step_top_logprob_increase_values"] = torch.stack(tracker.until_step_per_example_top_logprob_increase_values,
                                                                                          dim=1)
        token_metrics["per_example_until_step_top_prob_increase_token_ids"] = torch.stack(tracker.until_step_per_example_top_prob_increase_token_ids,
                                                                                          dim=1)
        token_metrics["per_example_until_step_top_prob_increase_values"] = torch.stack(tracker.until_step_per_example_top_prob_increase_values, dim=1)

        token_metrics["initial_hidden_representation_norms"] = tracker.initial_hidden_representations.norm(dim=1).cpu()
        token_metrics["final_hidden_representation_norms"] = tracker.prev_epoch_hidden_representations.norm(dim=1).cpu()
        token_metrics["hidden_representation_change_norms"] = torch.norm(tracker.initial_hidden_representations -
                                                                         tracker.prev_epoch_hidden_representations, dim=1).cpu()

        torch.save(token_metrics, os.path.join(experiment_dir, file_name))
