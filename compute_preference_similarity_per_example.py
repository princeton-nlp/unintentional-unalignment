import argparse
import os
from datetime import datetime

import datasets
import torch
from tqdm import tqdm

import common.utils.logging as logging_utils
from utils.script_utils import load_tokenizer_and_model

PADDING_TOKEN = "<|padding|>"
PROMPT_TOKEN = "<|prompter|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|endoftext|>"


def __orig_alpacafarm_create_format_input_func():
    def format_input_func(example):
        new_example = {}
        instruction, input_text = example["instruction"], example["input"]
        if input_text:
            query = f"{PROMPT_TOKEN}{instruction}\n{input_text}{ASSISTANT_TOKEN}"
        else:
            query = f"{PROMPT_TOKEN}{instruction}{ASSISTANT_TOKEN}"

        if example["preference"] == 1:
            selected = example["output_1"]
            rejected = example["output_2"]
        else:
            selected = example["output_2"]
            rejected = example["output_1"]

        new_example["query"] = query
        new_example["text_w"] = f"{query}{selected}"
        new_example["text_l"] = f"{query}{rejected}"
        return new_example

    return format_input_func


def __ultrafeedback_create_format_input_func():
    def format_input_func(example):
        new_example = {}
        chosen, rejected = example["chosen"], example["rejected"]
        query = f"{PROMPT_TOKEN}{chosen[0]['content']}{ASSISTANT_TOKEN}"

        new_example["query"] = query
        new_example["text_w"] = f"{query}{chosen[1]['content']}"
        new_example["text_l"] = f"{query}{rejected[1]['content']}"
        return new_example

    return format_input_func


def __create_chat_template_format_input_func(tokenizer, query_field: str = "query", chosen_field: str = "chosen", rejected_field: str = "rejected"):
    def format_input_func(example):
        new_example = {}

        query = [{"role": "user", "content": example[query_field]}]
        query = tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
        new_example["query"] = query
        new_example["text_w"] = f"{query}" + example[chosen_field]
        new_example["text_l"] = f"{query}" + example[rejected_field]
        return new_example

    return format_input_func


DATASET_CREATE_FORMAT_INPUT_FUNC = {
    "tatsu-lab/alpaca_farm": __orig_alpacafarm_create_format_input_func,
    "HuggingFaceH4/ultrafeedback_binarized": __ultrafeedback_create_format_input_func
}


def __get_dataset(dataset_name: str, cache_dir: str = None):
    if dataset_name == "tatsu-lab/alpaca_farm":
        return datasets.load_dataset(dataset_name, name="alpaca_human_preference", split="preference", cache_dir=cache_dir)
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        return datasets.load_dataset(dataset_name, split="train_prefs", cache_dir=cache_dir)
    else:
        # Loads dataset from JSON file for all other datasets
        return datasets.load_dataset("json", data_files=dataset_name, split="train")


def __subsample_dataset(dataset, num_train_samples: int = -1, train_samples_random_seed: int = -1):
    if num_train_samples < 0:
        return torch.arange(len(dataset)), dataset

    if train_samples_random_seed > 0:
        perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(train_samples_random_seed))
    else:
        perm = torch.randperm(len(dataset))

    num_samples = min(num_train_samples, len(dataset))
    sample_indices = perm[:num_samples]
    dataset = dataset.select(sample_indices)
    return sample_indices, dataset


def __prepare_and_tokenize_dataset(sample_indices, dataset_name, dataset, tokenizer, max_input_length: int,
                                   chat_query_field: str = "query", chat_chosen_field: str = "chosen", chat_rejected_field: str = "rejected"):
    if not tokenizer.chat_template:
        format_input_func = DATASET_CREATE_FORMAT_INPUT_FUNC[dataset_name]()
    else:
        format_input_func = __create_chat_template_format_input_func(tokenizer, query_field=chat_query_field,
                                                                     chosen_field=chat_chosen_field, rejected_field=chat_rejected_field)

    dataset = dataset.map(format_input_func, batched=False)
    dataset = dataset.select_columns(["query", "text_w", "text_l"])

    max_input_length = max_input_length if max_input_length > 0 else None

    def tokenize_examples(example: dict):
        query_input_ids = tokenizer(example["query"], padding=False, truncation=max_input_length is not None,
                                    max_length=max_input_length, return_tensors="pt", add_special_tokens=not tokenizer.chat_template).input_ids
        text_w_input_ids = tokenizer(example["text_w"], padding=False, truncation=max_input_length is not None,
                                     max_length=max_input_length, return_tensors="pt", add_special_tokens=not tokenizer.chat_template).input_ids
        text_l_input_ids = tokenizer(example["text_l"], padding=False, truncation=max_input_length is not None,
                                     max_length=max_input_length, return_tensors="pt", add_special_tokens=not tokenizer.chat_template).input_ids
        return {
            "query": query_input_ids,
            "text_w": text_w_input_ids,
            "text_l": text_l_input_ids
        }

    dataset = dataset.map(tokenize_examples, batched=False)
    dataset.set_format(type="torch")

    indices_to_include = []
    for i, example in enumerate(dataset):
        query_len = example["query"][0].shape[0]
        preferred_token_ids = example["text_w"][0][query_len:]
        dispreferred_token_ids = example["text_l"][0][query_len:]

        if query_len == 0 or preferred_token_ids.shape[0] == 0 or dispreferred_token_ids.shape[0] == 0:
            continue

        indices_to_include.append(i)

    dataset = dataset.select(indices_to_include)
    return sample_indices[indices_to_include], dataset


def __update_tokenizer_setting_and_chat_tokens(tokenizer):
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    if not tokenizer.eos_token:
        tokenizer.eos_token = EOS_TOKEN

    if not tokenizer.chat_template:
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({"pad_token": PADDING_TOKEN})
        tokenizer.add_special_tokens({"additional_special_tokens": [PROMPT_TOKEN, ASSISTANT_TOKEN]})
    else:
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token


# Taken from the torchaudio edit_distance function: https://pytorch.org/audio/main/generated/torchaudio.functional.edit_distance.html
def __normalized_edit_distance(seq1, seq2):
    len_sent2 = len(seq2)
    dold = list(range(len_sent2 + 1))
    dnew = [0 for _ in range(len_sent2 + 1)]

    for i in range(1, len(seq1) + 1):
        dnew[0] = i
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                substitution = dold[j - 1] + 1
                insertion = dnew[j - 1] + 1
                deletion = dold[j] + 1
                dnew[j] = min(substitution, insertion, deletion)

        dnew, dold = dold, dnew

    return int(dold[-1]) / max(len(seq1), len(seq2))


def get_and_log_preferred_dispreferred_normalized_edit_distance(logger, dataset):
    per_example_normalized_edit_distance = []

    logger.info(f"Starting normalized edit distance computation")
    for example in tqdm(dataset):
        query_len = example["query"][0].shape[0]
        preferred_token_ids = example["text_w"][0][query_len:]
        dispreferred_token_ids = example["text_l"][0][query_len:]

        if preferred_token_ids.shape[0] == 0 or dispreferred_token_ids.shape[0] == 0:
            raise ValueError("Found example with preferred or dispreferred outputs have length zero after truncating at maximal allowed length for "
                             "prompt + output.")

        pref_dispref_normalized_edit_distance = __normalized_edit_distance(preferred_token_ids, dispreferred_token_ids)
        per_example_normalized_edit_distance.append(pref_dispref_normalized_edit_distance)

    logger.info("\n===========================================================================================================================\n"
                "Edit distance metrics\n"
                "===========================================================================================================================")

    per_example_normalized_edit_distance = torch.tensor(per_example_normalized_edit_distance).to(torch.float)
    logger.info(f"\n------------------------------------------------------------------------------------------------------------------------------\n"
                f"Normalized edit distance of preferred and dispreferred outputs\n"
                f"Mean: {per_example_normalized_edit_distance.mean()} , Min: {per_example_normalized_edit_distance.min()} , "
                f"25th percentile: {torch.quantile(per_example_normalized_edit_distance, q=0.25)} , "
                f"Median: {per_example_normalized_edit_distance.median()} , "
                f"75th percentile: {torch.quantile(per_example_normalized_edit_distance, q=0.75)} , "
                f"Max: {per_example_normalized_edit_distance.max()}\n"
                f"------------------------------------------------------------------------------------------------------------------------------")

    return per_example_normalized_edit_distance


def __trim_padding(input_ids, tokenizer):
    return input_ids[torch.argmax((input_ids != tokenizer.vocab[tokenizer.eos_token]).to(torch.int)):]


def get_and_log_hidden_embedding_based_pref_similarities(logger, dataset, model, device, tokenizer):
    ches_scores = []
    ln_ches_scores = []
    last_hidden_embedding_inner_prods = []

    logger.info(f"Starting CHES scores and last hidden embedding inner products computation")
    model.to(device)
    for example in tqdm(dataset):
        query_len = __trim_padding(example["query"][0], tokenizer).shape[0]
        preferred = __trim_padding(example["text_w"][0], tokenizer).to(device)
        dispreferred = __trim_padding(example["text_l"][0], tokenizer).to(device)

        if query_len == 0:
            logger.warn("Skipping example since query length was zero tokens.")
            continue

        if query_len == preferred.shape[0] or query_len == dispreferred.shape[0]:
            logger.warn("Skipping example since preferred or dispreferred outputs have length zero after truncating at maximal allowed length for "
                        "prompt + output.")
            continue

        preferred_outputs = model(input_ids=preferred.unsqueeze(dim=0), output_hidden_states=True)
        preferred_hidden_embed = preferred_outputs.hidden_states[-1][0][query_len - 1:]
        dispreferred_outputs = model(input_ids=dispreferred.unsqueeze(dim=0), output_hidden_states=True)
        dispreferred_hidden_embed = dispreferred_outputs.hidden_states[-1][0][query_len - 1:]

        sum_preferred_embeddings = preferred_hidden_embed.sum(dim=0)
        sum_dispreferred_embeddings = dispreferred_hidden_embed.sum(dim=0)

        ches_score = (sum_preferred_embeddings * sum_dispreferred_embeddings).sum() - torch.norm(sum_preferred_embeddings) ** 2
        ches_scores.append(ches_score.cpu())

        preferred_length = preferred_hidden_embed.shape[0]
        dispreferred_length = dispreferred_hidden_embed.shape[0]
        pref_dispref = (sum_preferred_embeddings * sum_dispreferred_embeddings).sum() / (preferred_length * dispreferred_length)
        pref_only = torch.norm(sum_preferred_embeddings) ** 2 / (preferred_length ** 2)
        ln_ches_scores.append((pref_dispref - pref_only).cpu())

        last_hidden_embedding_inner_prods.append(torch.inner(preferred_hidden_embed[-1], dispreferred_hidden_embed[-1]).cpu())

    ches_scores = torch.tensor(ches_scores)
    ln_ches_scores = torch.tensor(ln_ches_scores)
    last_hidden_embedding_inner_prods = torch.tensor(last_hidden_embedding_inner_prods)

    logger.info(f"\n------------------------------------------------------------------------------------------------------------------------------\n"
                f"CHES Scores:\n"
                f"Mean: {ches_scores.mean()} , "
                f"Min: {ches_scores.min()} , "
                f"25th percentile: {torch.quantile(ches_scores, q=0.25)} , "
                f"Median: {ches_scores.median()} , "
                f"75th percentile: {torch.quantile(ches_scores, q=0.75)} , "
                f"Max: {ches_scores.max()}\n"
                f"------------------------------------------------------------------------------------------------------------------------------")

    logger.info(f"\n------------------------------------------------------------------------------------------------------------------------------\n"
                f"Length-Normalized CHES Scores:\n"
                f"Mean: {ln_ches_scores.mean()} , "
                f"Min: {ln_ches_scores.min()} , "
                f"25th percentile: {torch.quantile(ln_ches_scores, q=0.25)} , "
                f"Median: {ln_ches_scores.median()} , "
                f"75th percentile: {torch.quantile(ln_ches_scores, q=0.75)} , "
                f"Max: {ln_ches_scores.max()}\n"
                f"------------------------------------------------------------------------------------------------------------------------------")

    logger.info(f"\n------------------------------------------------------------------------------------------------------------------------------\n"
                f"Last Hidden Embedding Inner Products:\n"
                f"Mean: {last_hidden_embedding_inner_prods.mean()} , "
                f"Min: {last_hidden_embedding_inner_prods.min()} , "
                f"25th percentile: {torch.quantile(last_hidden_embedding_inner_prods, q=0.25)} , "
                f"Median: {last_hidden_embedding_inner_prods.median()} , "
                f"75th percentile: {torch.quantile(last_hidden_embedding_inner_prods, q=0.75)} , "
                f"Max: {last_hidden_embedding_inner_prods.max()}\n"
                f"------------------------------------------------------------------------------------------------------------------------------")

    return ches_scores, ln_ches_scores, last_hidden_embedding_inner_prods


@torch.no_grad()
def main(config: dict):
    model_name = config["model"]
    dataset_name = config["dataset"]
    num_train_samples = config["num_train_samples"]
    train_samples_random_seed = config["train_samples_random_seed"]
    max_input_length = config["max_input_length"]
    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() and config["gpu_id"] >= 0 else "cpu")

    dataset_display_name = config["custom_dataset_display_name"] if config["custom_dataset_display_name"] else dataset_name.split("/")[-1]
    subdir_name = model_name.split("/")[-1] + "_" + dataset_display_name
    logger = logging_utils.create_logger(file_logging=not config["dont_save_logs"],
                                         log_dir=os.path.join(config["output_dir"], subdir_name),
                                         log_file_name_prefix=f"log_samples_{num_train_samples}")
    logger.info(f"Config: {config}")

    try:
        start_time = datetime.utcnow()

        logger.info(f"======================================================================================================")
        logger.info(f"Model: '{model_name}', Dataset: '{dataset_name}'")
        logger.info(f"======================================================================================================\n")

        tokenizer, model = load_tokenizer_and_model(model_name, cache_dir=config["cache_dir"], device=device)
        model.to(device)
        __update_tokenizer_setting_and_chat_tokens(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

        dataset = __get_dataset(dataset_name, cache_dir=config["cache_dir"])
        sample_indices, dataset = __subsample_dataset(dataset, num_train_samples, train_samples_random_seed)
        sample_indices, tokenized_dataset = __prepare_and_tokenize_dataset(sample_indices, dataset_name, dataset, tokenizer, max_input_length,
                                                                           chat_chosen_field=config["chat_chosen_field"],
                                                                           chat_rejected_field=config["chat_rejected_field"])
        logger.info(f"Filtered out samples with empty query or outputs\n"
                    f"Original number of samples: {len(dataset)}\n"
                    f"Number of samples after filtering: {len(sample_indices)}")

        normalized_edit_distances = get_and_log_preferred_dispreferred_normalized_edit_distance(logger, tokenized_dataset)
        ches_scores, ln_ches_scores, last_hidden_embedding_inner_prods = get_and_log_hidden_embedding_based_pref_similarities(logger,
                                                                                                                              tokenized_dataset,
                                                                                                                              model,
                                                                                                                              device,
                                                                                                                              tokenizer)

        results = {
            "sample_indices": sample_indices,
            "minus_normalized_edit_distances": - normalized_edit_distances,
            "ches_scores": ches_scores,
            "ln_ches_scores": ln_ches_scores,
            "last_hidden_embedding_inner_prods": last_hidden_embedding_inner_prods
        }
        torch.save(results, os.path.join(config["output_dir"], subdir_name, f"results_samples.pt"))

        end_time = datetime.utcnow()
        logger.info(f"Finished script, time took: {end_time - start_time}")
    except Exception:
        logger.exception("Exception while running script.")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="outputs/pref_similarity", help="Directory to save log file to")
    p.add_argument("--cache_dir", type=str, default=None, help="Directory of cache for HuggingFace models and datasets")
    p.add_argument("--dont_save_logs", action="store_true", help="Only log to console, and not to a file")
    p.add_argument("--model", type=str, default="allenai/OLMo-1B-hf", help="Model to use")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca_farm", help="Dataset to use")
    p.add_argument("--custom_dataset_display_name", type=str, default="", help="Name of dataset to use for creating file name")
    p.add_argument("--num_train_samples", type=int, default=-1,
                   help="Number of training samples to compute preference similarity for (if < 0, all samples are used)")
    p.add_argument("--train_samples_random_seed", type=int, default=-1, help="Random seed to use for selecting train samples")
    p.add_argument("--max_input_length", type=int, default=512,
                   help="Truncate outputs to this maximal length (if < 0, does not truncate)")
    p.add_argument("--chat_chosen_field", type=str, default="chosen", help="Field name for chosen output when using models with chat template")
    p.add_argument("--chat_rejected_field", type=str, default="rejected", help="Field name for rejected output when using models with chat template")
    p.add_argument("--gpu_id", type=int, default=-1, help="GPU id to use (-1 for CPU)")
    args = p.parse_args()

    main(args.__dict__)
