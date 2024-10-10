import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset

import common.utils.logging as logging_utils
import utils.script_utils as script_utils
import utils.sorry_bench_utils as sorry_bench_utils


def __set_initial_random_seed(random_seed: int):
    if random_seed > 0:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


def __prepare_tokenizer(tokenizer):
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"


def __generate_train_and_test_responses(model_name, train_queries, test_queries, batch_size, cache_dir: str = None, device=torch.device("cpu")):
    tokenizer, model = script_utils.load_tokenizer_and_model(model_name=model_name, cache_dir=cache_dir, device=device)
    __prepare_tokenizer(tokenizer)

    train_responses = sorry_bench_utils.generate_responses_for_eval(tokenizer, model, queries=train_queries, batch_size=batch_size,
                                                                    logger=logging_utils, device=device)
    test_responses = sorry_bench_utils.generate_responses_for_eval(tokenizer, model, queries=test_queries, batch_size=batch_size,
                                                                  logger=logging_utils, device=device)
    return train_responses, test_responses


def __evaluate_refusal_rates(model_name, judge_model_name, train_dataset, test_dataset, batch_size, output_dir,
                             cache_dir: str = None, device=torch.device("cpu")):
    logging_utils.info(f"Evaluating refusal rates for: {model_name}")

    train_queries = [example["query"] for example in train_dataset]
    test_queries = [example["query"] for example in test_dataset]

    logging_utils.info("Generating responses for the training and test sets")
    train_responses, test_responses = __generate_train_and_test_responses(model_name, train_queries, test_queries,
                                                                        batch_size=batch_size, cache_dir=cache_dir, device=device)

    judge_tokenizer, judge_model = script_utils.load_tokenizer_and_model(model_name=judge_model_name,
                                                                         cache_dir=cache_dir,
                                                                         device=device)
    __prepare_tokenizer(judge_tokenizer)

    train_scores = sorry_bench_utils.get_judge_scores(judge_tokenizer, judge_model, train_queries, train_responses,
                                                      batch_size=batch_size,
                                                      device=device,
                                                      logger=logging_utils)
    test_scores = sorry_bench_utils.get_judge_scores(judge_tokenizer, judge_model, test_queries, test_responses,
                                                    batch_size=batch_size,
                                                    device=device,
                                                    logger=logging_utils)

    refusal_metrics = {
        "train_refusal_rate": 1 - sum(train_scores) / len(train_scores),
        "test_refusal_rate": 1 - sum(test_scores) / len(test_scores)
    }
    file_name = f"{model_name.split('/')[-1]}_refusal_eval.json"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(refusal_metrics, f, indent=2)

    logging_utils.info(f"Saved refusal rate metrics for {model_name} at: {os.path.join(output_dir, file_name)}\n"
                       f"{refusal_metrics}")

    full_responses = {
        "train": [{"query": query, "response": response, "score": score} for query, response, score in
                  zip(train_queries, train_responses, train_scores)],
        "test": [{"query": query, "response": response, "score": score} for query, response, score in
                       zip(test_queries, test_responses, test_scores)]
    }
    responses_file_name = f"{model_name.split('/')[-1]}_full_responses.json"
    with open(os.path.join(output_dir, responses_file_name), "w") as f:
        json.dump(full_responses, f, indent=2)

    logging_utils.info(f"Saved full responses for {model_name} at: {os.path.join(output_dir, responses_file_name)}")


def main(config: dict):
    __set_initial_random_seed(config["random_seed"])
    logging_utils.init_console_logging()

    subdir_name = f"results_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(config["output_dir"], subdir_name)

    logging_utils.init_file_logging(log_file_name_prefix="log", output_dir=output_dir)
    logging_utils.info(f"Evaluating refusal rates using the config:\n{config}")

    try:
        start_time = datetime.utcnow()

        train_dataset = load_dataset("json", data_files=os.path.join(config["dataset_path"], "train.jsonl"), split="train")
        test_dataset = load_dataset("json", data_files=os.path.join(config["dataset_path"], "test.jsonl"), split="train")

        device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() and config["gpu_id"] >= 0 else "cpu")

        for model_name in config["models"]:
            __evaluate_refusal_rates(model_name, config["judge_model"], train_dataset, test_dataset,
                                     batch_size=config["batch_size"],
                                     output_dir=output_dir,
                                     cache_dir=config["cache_dir"],
                                     device=device)

        end_time = datetime.utcnow()
        logging_utils.info(f"Finished script, time took: {end_time - start_time}")
    except Exception:
        logging_utils.exception("Exception while running script.")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_path", type=str, help="Path to directory with the created SORRY-Bench preference dataset files")
    p.add_argument("--random_seed", default=-1, type=int, help="Random seed to use (by default does not set a random seed)")
    p.add_argument("--output_dir", default="outputs/sorrybench_refusal_rates", help="Directory to save created data files in")
    p.add_argument("--models", nargs="+", type=str, default=["meta-llama/Meta-Llama-3-8B-Instruct"],
                   help="Models to evaluate refusal rates for")
    p.add_argument("--judge_model", type=str, default="sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",
                   help="Model to use for judging outputs")
    p.add_argument("--cache_dir", type=str, default=None, help="Cache directory for Hugging Face models")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for generating outputs and scoring them")
    p.add_argument("--gpu_id", type=int, default=-1, help="GPU id to use (-1 for CPU)")

    args = p.parse_args()

    main(args.__dict__)
