import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer


def __load_json(experiment_dir_path: Path, file_name: str):
    summary_json_path = experiment_dir_path.joinpath(file_name)

    if not summary_json_path.exists():
        return None

    with open(summary_json_path) as f:
        return json.load(f)


def __load_token_metrics(experiment_dir_path: Path):
    token_metrics_path = experiment_dir_path.joinpath("token_metrics.pt")

    if not token_metrics_path.exists():
        return None

    return torch.load(token_metrics_path, map_location="cpu")


def __load_aggregate_metrics_configs_and_summaries_from_exps_in_directory(experiments_dir: str, filter_runs_where_train_loss_increased: bool = False):
    aggregate_metrics = {
        "num_train_samples": 0,
        "per_example_per_step_is_pref_logprob_smaller_than_init": [],
        "per_example_per_step_did_preferred_prob_decrease": [],
        "per_example_per_step_did_train_loss_decrease": [],
        "per_example_initial_preferred_token_logprob": [],
        "per_example_initial_dispreferred_token_logprob": [],
        "per_example_per_step_preferred_token_logprob_change": [],
        "per_example_per_step_dispreferred_token_logprob_change": [],
        "per_example_until_step_preferred_token_logprob_increase_rank": [],
        "per_example_until_step_top_logprob_increase_token_ids": [],
        "per_example_until_step_top_logprob_increase_values": [],
        "per_example_until_step_preferred_token_prob_increase_rank": [],
        "per_example_until_step_top_prob_increase_token_ids": [],
        "per_example_until_step_top_prob_increase_values": [],
        "initial_preferred_token_norms": [],
        "initial_dispreferred_token_norms": [],
        "initial_preferred_and_dispreferred_inner_products": [],
        "initial_dispreferred_projection_coeffs": [],
        "initial_preferred_norm_dispreferred_projection_coeff_diffs": [],
        "initial_norm_orthogonal_component_of_dispreferred_tokens": [],
        "final_preferred_token_norms": [],
        "final_dispreferred_token_norms": [],
        "final_preferred_and_dispreferred_inner_products": [],
        "final_dispreferred_projection_coeffs": [],
        "final_preferred_norm_dispreferred_projection_coeff_diffs": [],
        "final_norm_orthogonal_component_of_dispreferred_tokens": [],
        "initial_hidden_representation_norms": [],
        "final_hidden_representation_norms": [],
        "hidden_representation_change_norms": []
    }
    configs = []
    summaries = []

    experiments_dir_path = Path(experiments_dir)
    experiment_paths = [path for path in experiments_dir_path.iterdir() if path.is_dir()]

    for experiment_path in experiment_paths:
        exp_summary = __load_json(experiment_path, file_name="summary.json")
        config = __load_json(experiment_path, file_name="config.json")
        token_metrics = __load_token_metrics(experiment_path)

        if not exp_summary or not token_metrics:
            print(f"Skipping experiment {experiment_path} as either summary or token metrics are missing.")
            continue

        per_step_did_train_loss_decrease = token_metrics["per_step_did_train_loss_decrease"]
        if filter_runs_where_train_loss_increased and torch.any(~per_step_did_train_loss_decrease):
            continue

        configs.append(config)
        summaries.append(exp_summary)
        aggregate_metrics["num_train_samples"] += exp_summary["num_train_samples"]

        for key, value in token_metrics.items():
            if key == "per_step_did_train_loss_decrease":
                aggregate_metrics["per_example_per_step_did_train_loss_decrease"].append(value.repeat(exp_summary["num_train_samples"], 1))
            else:
                aggregate_metrics[key].append(value)

    for key, value in aggregate_metrics.items():
        if key == "num_train_samples":
            continue
        aggregate_metrics[key] = torch.cat(value, dim=0)

    print(f"Filtered out {len(experiment_paths) - len(configs)} experiments in which the training loss increased at some time step")
    return aggregate_metrics, configs, summaries


def __print_experiment_info(experiments_dir: str, config: dict, summary: dict):
    print(f"========================================================================================================")
    print(f"Experiment info {experiments_dir}:")
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"Model: {config['model']}\n"
          f"Model checkpoint: {config['load_model_checkpoint']}\n"
          f"Dataset: {config['dataset']}\n"
          f"Preferred output token: '{summary['data']['preferred_output'][0]}'\n"
          f"Dispreferred output token: '{summary['data']['dispreferred_output'][0]}'")
    print(f"========================================================================================================")


def __find_max_decrease_steps(per_example_value_tensor: torch.Tensor):
    first_indices = []
    second_indices = []
    for i in range(per_example_value_tensor.shape[0]):
        value_tensor = per_example_value_tensor[i]
        all_diffs_mat = value_tensor.unsqueeze(dim=0) - value_tensor.unsqueeze(dim=1)

        upper_triangular_mask = torch.triu(torch.ones_like(all_diffs_mat), diagonal=1).bool()
        all_diffs_mat[upper_triangular_mask] = - torch.inf

        max_value = all_diffs_mat.max()
        max_index = (all_diffs_mat == max_value).nonzero(as_tuple=False)

        # Return the first occurrence of the maximum value's row and column indices
        second_index, first_index = max_index[0].tolist()
        first_indices.append(first_index)
        second_indices.append(second_index)

    return torch.tensor(first_indices, dtype=torch.int64), torch.tensor(second_indices, dtype=torch.int64)


def __print_preferred_output_probability_decrease_results(aggregate_metrics: dict, max_step: int = -1):
    max_step = max_step if max_step > 0 else aggregate_metrics["per_example_per_step_did_preferred_prob_decrease"].shape[1]

    num_train_samples = aggregate_metrics["num_train_samples"]
    per_example_per_step_did_preferred_prob_decrease = aggregate_metrics["per_example_per_step_did_preferred_prob_decrease"][:, :max_step]
    per_example_per_step_did_train_loss_decrease = aggregate_metrics["per_example_per_step_did_train_loss_decrease"][:, :max_step]
    per_example_per_step_did_preferred_prob_decrease_and_loss_decrease = torch.logical_and(per_example_per_step_did_preferred_prob_decrease,
                                                                                           per_example_per_step_did_train_loss_decrease)

    # Compute first step for each example in which the preferred probability decreased
    first_step_preferred_prob_decreased = per_example_per_step_did_preferred_prob_decrease.to(torch.int).argmax(dim=1)
    first_step_preferred_prob_decreased = first_step_preferred_prob_decreased[
        per_example_per_step_did_preferred_prob_decrease.to(torch.int).sum(dim=1) > 0
        ]
    if first_step_preferred_prob_decreased.numel() == 0:
        first_step_preferred_prob_decreased = torch.tensor(-1)

    print(f"========================================================================================================")
    print(f"Preferred output probability decrease results:")
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"First step preferred output probability decreased: Min: {first_step_preferred_prob_decreased.min()} ,"
          f" Max: {first_step_preferred_prob_decreased.max()} , Median: {first_step_preferred_prob_decreased.median()} , "
          f"Mean: {first_step_preferred_prob_decreased.float().mean()}")
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"Number of examples in which preferred output probability decreased: "
          f"{(per_example_per_step_did_preferred_prob_decrease.sum(dim=1) > 0).sum()} / {num_train_samples}")
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"Number of examples in which preferred output probability decreased and train loss decreased: "
          f"{(per_example_per_step_did_preferred_prob_decrease_and_loss_decrease.sum(dim=1) > 0).sum()} / {num_train_samples}")
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"Number of examples in which train loss INCREASED: "
          f"{((~per_example_per_step_did_train_loss_decrease).sum(dim=1) > 0).sum()} / {num_train_samples}")
    print(f"========================================================================================================")

    # Print max decrease metrics
    initial_preferred_logprobs = aggregate_metrics["per_example_initial_preferred_token_logprob"]
    per_example_per_step_preferred_token_logprob_change = aggregate_metrics["per_example_per_step_preferred_token_logprob_change"][:, :max_step]
    per_step_curr_preferred_logprobs = per_example_per_step_preferred_token_logprob_change + initial_preferred_logprobs.unsqueeze(dim=1)

    per_step_including_initial_logprobs = torch.cat([initial_preferred_logprobs.unsqueeze(dim=1), per_step_curr_preferred_logprobs], dim=1)
    logprob_max_decrease_start_before_steps, logprob_max_decrease_end_before_steps = __find_max_decrease_steps(per_step_including_initial_logprobs)
    logprob_max_decrease_start_values = per_step_including_initial_logprobs[
        torch.arange(per_step_including_initial_logprobs.shape[0]), logprob_max_decrease_start_before_steps
    ]
    logprob_max_decrease_end_values = per_step_including_initial_logprobs[
        torch.arange(per_step_including_initial_logprobs.shape[0]), logprob_max_decrease_end_before_steps
    ]

    per_step_including_initial_probs = torch.exp(per_step_including_initial_logprobs)
    prob_max_decrease_start_before_steps, prob_max_decrease_end_before_steps = __find_max_decrease_steps(per_step_including_initial_probs)
    prob_max_decrease_start_values = per_step_including_initial_probs[
        torch.arange(per_step_including_initial_probs.shape[0]), prob_max_decrease_start_before_steps
    ]
    prob_max_decrease_end_values = per_step_including_initial_probs[
        torch.arange(per_step_including_initial_probs.shape[0]), prob_max_decrease_end_before_steps
    ]

    print()
    print(f"========================================================================================================")
    print(f"Max preferred output probability and log probability decrease results:")
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"Mean largest preferred output log probability decrease (step refers to value before that step): "
          f"{(logprob_max_decrease_end_values - logprob_max_decrease_start_values).mean()} ({logprob_max_decrease_start_values.mean()} -> {logprob_max_decrease_end_values.mean()})\n"
          f"Mean start step: {logprob_max_decrease_start_before_steps.float().mean()} , Mean end step: {logprob_max_decrease_end_before_steps.float().mean()}")
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"Mean largest preferred output probability decrease (step refers to value before that step): "
          f"{(prob_max_decrease_end_values - prob_max_decrease_start_values).mean()} ({prob_max_decrease_start_values.mean()} -> {prob_max_decrease_end_values.mean()})\n"
          f"Mean start step: {prob_max_decrease_start_before_steps.float().mean()} , Mean end step: {prob_max_decrease_end_before_steps.float().mean()}")
    print(f"========================================================================================================")

    print()
    print(f"========================================================================================================")
    print(f"Per example max preferred output probability and log probability decrease results")
    print(f"========================================================================================================")
    for i in range(logprob_max_decrease_start_before_steps.shape[0]):
        print(f"Example {i}")
        print(f"--------------------------------------------------------------------------------------------------------")
        print(f"Largest preferred output log probability decrease (step refers to value before that step):\n"
              f"Step {logprob_max_decrease_start_before_steps[i]} to {logprob_max_decrease_end_before_steps[i]}: "
              f"{logprob_max_decrease_end_values[i] - logprob_max_decrease_start_values[i]} "
              f"({logprob_max_decrease_start_values[i]} -> {logprob_max_decrease_end_values[i]})")
        print(f"--------------------------------------------------------------------------------------------------------")
        print(f"Largest preferred output probability decrease (step refers to value before that step):\n"
              f"Step {prob_max_decrease_start_before_steps[i]} to {prob_max_decrease_end_before_steps[i]}: "
              f"{prob_max_decrease_end_values[i] - prob_max_decrease_start_values[i]} "
              f"({prob_max_decrease_start_values[i]} -> {prob_max_decrease_end_values[i]})")
        print(f"========================================================================================================")
    print()


def __print_top_k_tokens_results(tokenizer, aggregate_metrics: dict, steps: List[int], top_k: List[int], num_tokens_to_print: int = 20):
    num_train_samples = aggregate_metrics["num_train_samples"]

    per_example_until_step_top_prob_increase_token_ids = aggregate_metrics["per_example_until_step_top_prob_increase_token_ids"]
    per_example_until_step_top_prob_increase_values = aggregate_metrics["per_example_until_step_top_prob_increase_values"]
    print(f"========================================================================================================")
    print(f"Tokens increasing most overall in probability:")
    for step in steps:
        for k in top_k:
            per_example_top_prob_increase_token_ids = per_example_until_step_top_prob_increase_token_ids[:, step - 1, :k]
            per_example_top_prob_increase_values = per_example_until_step_top_prob_increase_values[:, step - 1, :k]
            token_ids, counts = torch.unique(per_example_top_prob_increase_token_ids, return_counts=True)
            sort_by_counts_order = torch.argsort(counts, descending=True)
            token_ids = token_ids[sort_by_counts_order]
            counts = counts[sort_by_counts_order]

            print(f"--------------------------------------------------------------------------------------------------------")
            print(f"Overall Top-{k} probability increase token counts at time step {step}:")
            print(f"--------------------------------------------------------------------------------------------------------")
            for i in range(min(num_tokens_to_print, len(token_ids))):
                token_id = token_ids[i]
                count = counts[i]
                frequency = count.float() / num_train_samples

                token = tokenizer.convert_ids_to_tokens(token_id.view(1, -1))[0]
                decoded_token = tokenizer.decode(token_id)
                mean_increase = per_example_top_prob_increase_values[per_example_top_prob_increase_token_ids == token_id].mean()
                print(f"{token} (dec: {repr(decoded_token)}): count: {count} , frequency: {frequency:.3f} , mean increase: {mean_increase}")
            print()
    print(f"========================================================================================================")


def __print_token_unembedding_and_hidden_representation_metrics(aggregate_metrics: dict):
    print(f"========================================================================================================")
    print(f"Token unembeddings and hidden representation metrics:")
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"Preferred token norm: Initial: {aggregate_metrics['initial_preferred_token_norms'][0]} , "
          f"Final (mean): {aggregate_metrics['final_preferred_token_norms'].mean()}\n"
          f"Dispreferred token norm: Initial: {aggregate_metrics['initial_dispreferred_token_norms'][0]} , "
          f"Final (mean): {aggregate_metrics['final_dispreferred_token_norms'].mean()}\n"
          f"Preferred and dispreferred inner product: Initial {aggregate_metrics['initial_preferred_and_dispreferred_inner_products'][0]} , "
          f"Final (mean): {aggregate_metrics['final_preferred_and_dispreferred_inner_products'].mean()}\n"
          f"Dispreferred projection norm: Initial: {aggregate_metrics['initial_dispreferred_projection_coeffs'][0]} , "
          f"Final (mean): {aggregate_metrics['final_dispreferred_projection_coeffs'].mean()}\n"
          f"Preferred norm dispreferred projection norm diff: Initial: {aggregate_metrics['initial_preferred_norm_dispreferred_projection_coeff_diffs'][0]} , "
          f"Final (mean): {aggregate_metrics['final_preferred_norm_dispreferred_projection_coeff_diffs'].mean()}\n"
          f"Norm orthogonal component of dispreferred tokens: Initial: {aggregate_metrics['initial_norm_orthogonal_component_of_dispreferred_tokens'][0]} , "
          f"Final (mean): {aggregate_metrics['final_norm_orthogonal_component_of_dispreferred_tokens'].mean()}\n"
          f"Hidden representation norm mean: Initial: {aggregate_metrics['initial_hidden_representation_norms'].mean()} , "
          f"Final: {aggregate_metrics['final_hidden_representation_norms'].mean()}\n"
          f"Hidden representation change norm mean: {aggregate_metrics['hidden_representation_change_norms'].mean()}")


def print_results(experiments_dir: str, max_step: int = -1, filter_runs_where_train_loss_increased: bool = False):
    aggregate_metrics, configs, summaries = __load_aggregate_metrics_configs_and_summaries_from_exps_in_directory(experiments_dir,
                                                                                                                  filter_runs_where_train_loss_increased=filter_runs_where_train_loss_increased)

    __print_experiment_info(experiments_dir, configs[0], summaries[0])
    print()
    __print_preferred_output_probability_decrease_results(aggregate_metrics, max_step=max_step)
    print()
    tokenizer = AutoTokenizer.from_pretrained(configs[0]["model"])
    __print_top_k_tokens_results(tokenizer, aggregate_metrics, steps=[1, 5, 10, 25, 50, 100], top_k=[1, 2, 3])
    print()
    __print_token_unembedding_and_hidden_representation_metrics(aggregate_metrics)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments_dir", type=str, help="Directory to load experiments from")
    args = p.parse_args()

    print_results(args.experiments_dir, max_step=100, filter_runs_where_train_loss_increased=True)


if __name__ == "__main__":
    main()
