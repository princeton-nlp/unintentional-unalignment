import argparse
import os
import random
from collections import defaultdict
from datetime import datetime

import datasets
import numpy as np
import torch
from datasets import Dataset, load_dataset, DatasetDict

import common.utils.logging as logging_utils
from utils.script_utils import load_tokenizer_and_model
from utils.sorry_bench_utils import get_judge_scores

SORRYBENCH_DATASET = "sorry-bench/sorry-bench-202406"
SORRYBENCH_HUMAN_PREFERENCES_DATASET = "sorry-bench/sorry-bench-human-judgment-202406"


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


def __created_dataset_with_generated_responses(dataset, model_name, batch_size, generation_kwargs, cache_dir: str = None,
                                               device=torch.device("cpu")):
    tokenizer, gen_model = load_tokenizer_and_model(model_name=model_name, cache_dir=cache_dir, device=device)
    __prepare_tokenizer(tokenizer)

    new_examples = []
    num_batches = len(dataset) // batch_size
    if len(dataset) % batch_size != 0:
        num_batches += 1

    for i in range(0, len(dataset), batch_size):
        logging_utils.info(f"Generating responses for batch {i // batch_size + 1} / {num_batches}")
        batch = dataset[i:i + batch_size]
        queries = [turns_prompt[0] for turns_prompt in batch["turns"]]

        if tokenizer.chat_template:
            chats = [[{"role": "user", "content": query}] for query in queries]
            inputs = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]
            inputs = tokenizer(inputs, add_special_tokens=False, padding=True, return_tensors="pt").to(device)
        else:
            inputs = tokenizer(queries, padding=True, return_tensors="pt").to(device)

        # Generate two responses for each example in the batch
        outputs1 = gen_model.generate(**inputs, **generation_kwargs)
        outputs2 = gen_model.generate(**inputs, **generation_kwargs)

        # Decode the generated responses
        responses1 = tokenizer.batch_decode(outputs1[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses2 = tokenizer.batch_decode(outputs2[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Add the responses to the examples
        for j in range(len(responses1)):
            new_example = {key: values_list[j] for key, values_list in batch.items() if key != "turns"}
            new_example["query"] = queries[j]
            new_example["output1"] = responses1[j]
            new_example["output2"] = responses2[j]
            new_examples.append(new_example)

    return Dataset.from_list(new_examples)


def __populate_preferences_randomly(new_example, example, first_score, second_score):
    rejected = int(random.random() < 0.5) + 1
    chosen = 2 if rejected == 1 else 1
    new_example["rejected"] = example[f"output{rejected}"]
    new_example["rejected_score"] = first_score if rejected == 1 else second_score
    new_example["chosen"] = example[f"output{chosen}"]
    new_example["chosen_score"] = second_score if rejected == 1 else second_score


def __create_labeled_preferences_dataset(gen_preferences_dataset, judge_model_name, batch_size, cache_dir: str = None,
                                         device=torch.device("cpu")):
    tokenizer, judge_model = load_tokenizer_and_model(model_name=judge_model_name, cache_dir=cache_dir, device=device)
    __prepare_tokenizer(tokenizer)

    queries = [example["query"] for example in gen_preferences_dataset]
    first_responses = [example["output1"] for example in gen_preferences_dataset]
    second_responses = [example["output2"] for example in gen_preferences_dataset]

    first_responses_judge_scores = get_judge_scores(tokenizer, judge_model, queries, first_responses, batch_size, device, logger=logging_utils)
    second_responses_judge_scores = get_judge_scores(tokenizer, judge_model, queries, second_responses, batch_size, device, logger=logging_utils)

    new_examples = []
    for example, first_score, second_score in zip(gen_preferences_dataset, first_responses_judge_scores, second_responses_judge_scores):
        new_example = {key: value for key, value in example.items() if key not in ["output1", "output2"]}

        if first_score is None or second_score is None:
            logging_utils.warning(f"Encountered None score for example, Setting preferences randomly.\n"
                                  f"Example: {example}\n"
                                  f"First response score: {first_score}\n"
                                  f"Second response score: {second_score}")
            __populate_preferences_randomly(new_example, example, first_score, second_score)
        elif first_score > second_score:
            new_example["rejected"] = example["output1"]
            new_example["rejected_score"] = first_score
            new_example["chosen"] = example["output2"]
            new_example["chosen_score"] = second_score
        elif first_score < second_score:
            new_example["rejected"] = example["output2"]
            new_example["rejected_score"] = second_score
            new_example["chosen"] = example["output1"]
            new_example["chosen_score"] = first_score
        else:
            __populate_preferences_randomly(new_example, example, first_score, second_score)

        new_examples.append(new_example)

    return Dataset.from_list(new_examples)


def __relabel_ties_using_reward_model(gen_preferences_dataset, batch_size: int = 8, cache_dir: str = None, device=torch.device("cpu")):
    ties_indices = []
    queries = []
    first_responses = []
    second_responses = []

    for i, example in enumerate(gen_preferences_dataset):
        if example["chosen_score"] == example["rejected_score"]:
            ties_indices.append(i)
            queries.append(example["query"])
            first_responses.append(example["chosen"])
            second_responses.append(example["rejected"])

    logging_utils.info(f"Breaking preference ties using PairRM for {len(ties_indices)} examples")

    import utils.pairrm_utils as pairrm_utils
    comparison_results = pairrm_utils.get_comparison_results(queries, first_responses, second_responses,
                                                             batch_size=batch_size,
                                                             cache_dir=cache_dir,
                                                             device=device,
                                                             logger=logging_utils)

    new_examples = []
    for i, example in enumerate(gen_preferences_dataset):
        if i == ties_indices[0]:
            example["chosen"] = first_responses[0] if comparison_results[0] else second_responses[0]
            example["rejected"] = second_responses[0] if comparison_results[0] else first_responses[0]
            new_examples.append(example)

            ties_indices = ties_indices[1:]
            first_responses = first_responses[1:]
            second_responses = second_responses[1:]
            comparison_results = comparison_results[1:]
        else:
            new_examples.append(example)

    return Dataset.from_list(new_examples)


def __add_gold_refusal_and_non_refusal_outputs(gen_preferences_dataset, cache_dir: str = None):
    human_pref_dataset = datasets.load_dataset(SORRYBENCH_HUMAN_PREFERENCES_DATASET, cache_dir=cache_dir)
    human_pref_dataset = datasets.concatenate_datasets([human_pref_dataset["train"], human_pref_dataset["test"]])

    # Create dictionary of question id to refusal and non-refusal outputs
    question_id_to_outputs = defaultdict(lambda: {"refusal": [], "non_refusal": []})
    for example in human_pref_dataset:
        output_type = "refusal" if example["human_score"] < 0.5 else "non_refusal"
        question_id_to_outputs[example["question_id"]][output_type].append(example["choices"][0]["turns"][0])

    before_len = len(gen_preferences_dataset)
    gen_preferences_dataset = gen_preferences_dataset.filter(lambda example: len(question_id_to_outputs[example["question_id"]]["refusal"]) > 0 and
                                                                             len(question_id_to_outputs[example["question_id"]]["non_refusal"]) > 0)
    logging_utils.info(f"Filtered out {before_len - len(gen_preferences_dataset)} examples that do not have a gold refusal or non-refusal output")

    def add_gold_refusal_and_non_refusal(example):
        example["gold_refusal"] = random.choice(question_id_to_outputs[example["question_id"]]["refusal"])
        example["gold_non_refusal"] = random.choice(question_id_to_outputs[example["question_id"]]["non_refusal"])
        return example

    return gen_preferences_dataset.map(add_gold_refusal_and_non_refusal, batched=False)


def __log_dataset_stats(dataset, split: str = None):
    num_both_score_zero = 0
    num_both_score_one = 0
    num_different_scores = 0
    num_score_that_is_not_zero_or_one = 0
    for example in dataset:
        if example["chosen_score"] != example["rejected_score"]:
            num_different_scores += 1
        if example["chosen_score"] not in [0, 1] or example["rejected_score"] not in [0, 1]:
            num_score_that_is_not_zero_or_one += 1
        if example["chosen_score"] == 0 and example["rejected_score"] == 0:
            num_both_score_zero += 1
        if example["chosen_score"] == 1 and example["rejected_score"] == 1:
            num_both_score_one += 1

    dataset_split_name = "Dataset" if not split else f"{split} dataset"
    logging_utils.info(f"\n=============================================================================\n"
                       f"{dataset_split_name} stats\n"
                       f"=============================================================================\n"
                       f"Number of examples with both scores 0: {num_both_score_zero}\n"
                       f"Number of examples with both scores 1: {num_both_score_one}\n"
                       f"Number of examples with different scores: {num_different_scores}\n"
                       f"Number of examples with scores that are not 0 or 1: {num_score_that_is_not_zero_or_one}\n"
                       f"Total number of example: {len(dataset)}")

    log_msg = "First 5 examples from the dataset:\n"
    for i, example in enumerate(dataset.select(range(5))):
        log_msg += "------------------------------------------------------------------------\n"
        log_msg += f"{i + 1}. {example}\n"
        log_msg += "------------------------------------------------------------------------\n"
    logging_utils.info(log_msg)


def main(config: dict):
    __set_initial_random_seed(config["random_seed"])
    logging_utils.init_console_logging()

    use_rm_str = "_use_pairrm" if config["use_rm"] else ""
    subdir_name = (f"{config['model'].split('/')[-1]}_seed_{config['random_seed']}_temp_{str(config['temperature']).replace('.', '-')}_"
                   f"topp_{str(config['top_p']).replace('.', '-')}_test_{str(config['test_ratio']).replace('.', '-')}"
                   f"{use_rm_str}")
    subdir_name += f"_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(config["output_dir"], subdir_name)

    logging_utils.init_file_logging(log_file_name_prefix="log", output_dir=output_dir)
    logging_utils.info(f"Creating generated preference dataset using the config:\n{config}")

    try:
        start_time = datetime.utcnow()

        dataset = load_dataset(SORRYBENCH_DATASET, cache_dir=config["cache_dir"])["train"]
        dataset = dataset.filter(lambda example: example["prompt_style"] in config["prompt_styles"])

        device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() and config["gpu_id"] >= 0 else "cpu")
        generation_kwargs = {
            "top_k": 0.0,
            "top_p": config["top_p"],
            "do_sample": True,
            "max_new_tokens": config["max_output_length"],
            "temperature": config["temperature"],
            "use_cache": True
        }
        gen_preferences_dataset = __created_dataset_with_generated_responses(dataset,
                                                                             model_name=config["model"],
                                                                             batch_size=config["batch_size"],
                                                                             generation_kwargs=generation_kwargs,
                                                                             cache_dir=config["cache_dir"],
                                                                             device=device)

        gen_preferences_dataset = __create_labeled_preferences_dataset(gen_preferences_dataset,
                                                                       judge_model_name=config["judge_model"],
                                                                       batch_size=config["batch_size"],
                                                                       cache_dir=config["cache_dir"],
                                                                       device=device)

        if config["use_rm"]:
            gen_preferences_dataset = __relabel_ties_using_reward_model(gen_preferences_dataset,
                                                                        batch_size=config["rm_batch_size"],
                                                                        cache_dir=config["cache_dir"],
                                                                        device=device)

        gen_preferences_dataset = __add_gold_refusal_and_non_refusal_outputs(gen_preferences_dataset, cache_dir=config["cache_dir"])
        logging_utils.info("Added gold refusal and non-refusal outputs to the dataset")

        perm = torch.randperm(len(gen_preferences_dataset))
        n_test = int(len(perm) * config["test_ratio"])

        train_dataset = gen_preferences_dataset.select(perm[n_test:])
        test_dataset = gen_preferences_dataset.select(perm[:n_test])
        logging_utils.info(f"Number of training examples: {len(train_dataset)} , Number of test examples: {len(test_dataset)}")

        final_dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

        __log_dataset_stats(train_dataset, split="Train")
        __log_dataset_stats(test_dataset, split="Test")

        os.makedirs(output_dir, exist_ok=True)
        for split, dataset in final_dataset.items():
            dataset.to_json(os.path.join(output_dir, f"{split}.jsonl"))

        end_time = datetime.utcnow()
        logging_utils.info(f"Finished script, time took: {end_time - start_time}")
    except Exception:
        logging_utils.exception("Exception while running script.")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--random_seed", default=128, type=int, help="Random seed to use (if < 0 then no seed is set)")
    p.add_argument("--output_dir", default="data_files/sorrybench", help="Directory to save created data files in")
    p.add_argument("--prompt_styles", nargs="+", default=["base"], help="Prompt styles from the SorryBench dataset to include")
    p.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use for generating outputs")
    p.add_argument("--judge_model", type=str, default="sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",
                   help="Model to use for judging outputs")
    p.add_argument("--use_rm", action="store_true",
                   help="Use PairRM reward model for choosing preferences in case both are refusals or non-refusals")
    p.add_argument("--cache_dir", type=str, default=None, help="Cache directory for Hugging Face models")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for generating outputs and scoring them")
    p.add_argument("--rm_batch_size", type=int, default=4, help="Batch size for the reward model (only relevant when 'use_rm' is True")
    p.add_argument("--max_output_length", type=int, default=512, help="Maximal number of output tokens")
    p.add_argument("--top_p", type=float, default=1, help="Top-p sampling probability")
    p.add_argument("--temperature", type=float, default=1, help="Temperature for sampling")
    p.add_argument("--test_ratio", default=0.15, type=float, help="Ratio of training examples to keep for test")
    p.add_argument("--gpu_id", type=int, default=-1, help="GPU id to use (-1 for CPU)")

    args = p.parse_args()

    main(args.__dict__)
