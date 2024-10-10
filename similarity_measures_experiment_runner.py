import json
import os

os.environ["WANDB_SERVICE_WAIT"] = "10000"
os.environ["WANDB_INIT_TIMEOUT"] = "10000"
os.environ["WANDB_START_METHOD"] = "thread"
import wandb
import warnings
import logging
import jsonlines
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter("ignore")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from common_dpo.trainers.dpo_trainer import DPOTrainer
from common_dpo.trainers.dpo_config import DPOConfig
import torch
from absl import flags, app
import os
import accelerate
import datetime
import numpy as np
from tqdm import tqdm
import re
from collections import defaultdict
from common_dpo.trainers.utils import logprobs_from_logits, set_seed
import common.utils.logging as logging_utils

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", -1, "the random seed to set at beginning of training")
flags.DEFINE_integer("random_seed_for_training_example_selection", -1, "random seed used for ensuring same training example selection")
flags.DEFINE_string("wandb_project", "common_dpo", "the wandb project name")
flags.DEFINE_string("run_name", "run", "the wandb run name")
flags.DEFINE_string("output_dir", "outputs", "the output directory")
flags.DEFINE_string("preference_dataset_path", "HuggingFaceH4/ultrafeedback_binarized", "the path to the preference dataset")
flags.DEFINE_string("preference_similarity_path", "", "path to file with preference similarity data")
flags.DEFINE_float("use_samples_with_preference_similarity_quantile", 1,
                   "will use training samples whose preference similarity (in terms of hidden representations inner products or token overlap)"
                   " is centered around the given quantile")
flags.DEFINE_string("pref_similarity_measure", "ches_scores",
                    "the measure to use for choosing training samples according to preference similarity "
                    "(supports 'ches_scores', 'minus_normalized_edit_distances', and 'last_hidden_embedding_inner_prods'")
flags.DEFINE_integer("preference_num_samples", 512, "the number of samples to use from the preference dataset")
flags.DEFINE_integer("batch_size_per_gpu_proc", 4, "the batch size per gpu process (with accelerate the true batch size will be "
                                                   "this value * num_gpus used, without accelerate this will be the batch size)")
flags.DEFINE_integer("gradient_accumulation_steps", 1, "the gradient accumulation steps")
flags.DEFINE_integer("num_proc", 4, "Number of processes to use when preprocessing dataset")

flags.DEFINE_string("model", "allenai/OLMo-1B", "the path to/name of the pretrained model")
flags.DEFINE_string("cache_dir", None, "the cache directory")
flags.DEFINE_integer("max_input_length", 512, "the maximum input length")
flags.DEFINE_string("optimizer_name", "rmsprop", "name of optimizer to use")
flags.DEFINE_float("learning_rate", 1e-7, "the learning rate")

flags.DEFINE_integer("num_train_epochs", 1, "the number of training epochs")
flags.DEFINE_integer("eval_every_epochs", 1, "how often to evaluate")
flags.DEFINE_integer("save_every_epochs", -1, "how often to save checkpoints")
flags.DEFINE_bool("save_at_end", False, "save checkpoint at the end of training")
flags.DEFINE_integer("num_eval_generations_per_prompt", 1, "the number of generations per prompt during evaluation")

flags.DEFINE_string("objective", "dpo", "which objective to use (supports 'dpo', 'ipo', 'simpo', 'slic', and 'cross_entropy')")
flags.DEFINE_float("kl_coeff", 0.1, "the KL coefficient to use in the DPO and IPO objectives")
flags.DEFINE_float("sft_coeff", 0,
                   "if non-zero, will use an sft term in addition to dpo and ipo with this coefficient (e.g., as done in CPO, RPO, ORPO)")

flags.DEFINE_bool("model_parallel_without_accelerate", False,
                  "whether to use model parallelism (only use when running script without accelerate)")

PADDING_TOKEN = "<|padding|>"
PROMPT_TOKEN = "<|prompter|>"
ASSISTANT_TOKEN = "<|assistant|>"


def __aggregate_training_batch_stats(train_batch_stats):
    epoch_stats = {}
    metric_names = train_batch_stats[0].keys()
    for metric_name in metric_names:
        epoch_stats[metric_name] = np.mean([batch_stats[metric_name] for batch_stats in train_batch_stats])

    return epoch_stats


@torch.no_grad()
def __train_process_pref_batch(pref_batch, tokenizer, trainer, use_orig_text_l: bool = False):
    text_l = pref_batch["text_l"] if not use_orig_text_l else pref_batch["orig_text_l"]
    all_pref = pref_batch["text_w"] + text_l
    tokenized_inputs = tokenizer(all_pref, padding=True, truncation=True, max_length=FLAGS.max_input_length, return_tensors="pt")
    tokenized_input_ids = accelerate.utils.send_to_device(tokenized_inputs.input_ids, trainer.accelerator.device)
    attention_masks = accelerate.utils.send_to_device(tokenized_inputs.attention_mask, trainer.accelerator.device)

    input_ids_w = tokenized_input_ids[:len(pref_batch["text_w"])]
    attention_mask_w = attention_masks[:len(pref_batch["text_w"])]
    input_ids_l = tokenized_input_ids[len(pref_batch["text_w"]):]
    attention_mask_l = attention_masks[len(pref_batch["text_w"]):]

    return input_ids_w, attention_mask_w, input_ids_l, attention_mask_l


@torch.no_grad()
def __eval_process_input_ids(input_ids, attention_mask, trainer):
    input_data = {"input_ids": input_ids, "attention_mask": attention_mask}

    output = trainer.model(**input_data, output_hidden_states=False)
    logits = output.logits

    logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

    old_logits = trainer.ref_model(**input_data).logits
    old_logprobs = logprobs_from_logits(old_logits[:, :-1, :], input_ids[:, 1:])

    return logprobs, old_logprobs


@torch.no_grad()
def __generation_eval_process_batch(batch, tokenizer, trainer, generation_kwargs):
    #### Construct query tensors
    query_tensors = tokenizer(batch["query"], padding=True, truncation=True, max_length=128, return_tensors="pt")
    query_tensors = accelerate.utils.send_to_device(query_tensors, trainer.accelerator.device)

    #### Get generations from the model
    all_generation_tokens = []
    for i in range(FLAGS.num_eval_generations_per_prompt):  # generate multiple completions per prompt
        generation_tokens = trainer.accelerator.unwrap_model(trainer.model).generate(**query_tensors, **generation_kwargs)
        all_generation_tokens.append(generation_tokens)

    all_generation_tokens = torch.cat(all_generation_tokens, dim=0)

    attention_mask = torch.ones_like(all_generation_tokens)
    attention_mask[all_generation_tokens == tokenizer.pad_token_id] = 0
    logprobs, old_logprobs, = __eval_process_input_ids(all_generation_tokens,
                                                       attention_mask=attention_mask,
                                                       trainer=trainer)

    texts = tokenizer.batch_decode(all_generation_tokens, skip_special_tokens=True)

    #### Update batch with response
    batch["response"] = [x.split(ASSISTANT_TOKEN)[-1] for x in texts]
    batch["query"] = batch["query"] * FLAGS.num_eval_generations_per_prompt

    return batch, logprobs, old_logprobs


def __eval_generations(epoch, trainer, tokenizer, stats):
    log_main_process(trainer.accelerator, f"Computing generation evaluation on a single training batch")

    generation_kwargs = {
        "top_k": 0.0,  # no top-k sampling
        "top_p": 1.0,  # no nucleus sampling
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 256,  # specify how many tokens you want to generate at most
        "temperature": 1.0,  # control the temperature of the softmax, 1.0 means no change, lower means more greedy, higher means more diverse
        "use_cache": True,  # whether or not the model should use the past key/values attentions (if the model supports it)
    }

    eval_batch = next(iter(trainer.dataloader))
    eval_batch, logprobs, old_logprobs, = __generation_eval_process_batch(eval_batch,
                                                                          tokenizer,
                                                                          trainer,
                                                                          generation_kwargs)
    all_to_log = {}
    columns_to_log_eval = ["query", "response"]
    for k in columns_to_log_eval:
        all_to_log[k] = (
                all_to_log.get(k, []) + [x.cpu().numpy().item() if isinstance(x, torch.Tensor) else x for x in eval_batch[k]]
        )
    del eval_batch

    token_lengths = np.array([len(tokenizer(x).input_ids) for x in all_to_log["response"]])
    stats[f"eval_gen/train/token_lengths"] = token_lengths
    stats[f"eval_gen/train/token_lengths_mean"] = token_lengths.mean()
    stats[f"eval_gen/train/token_lengths_std"] = token_lengths.std()
    stats[f"eval_gen/train/token_lengths_max"] = token_lengths.max()
    stats[f"eval_gen/train/token_lengths_min"] = token_lengths.min()

    word_lengths = np.array([len(re.findall("[a-zA-Z_]+", x)) for x in all_to_log["response"]])
    stats[f"eval_gen/train/word_lengths"] = word_lengths
    stats[f"eval_gen/train/word_lengths_mean"] = word_lengths.mean()
    stats[f"eval_gen/train/word_lengths_std"] = word_lengths.std()
    stats[f"eval_gen/train/word_lengths_max"] = word_lengths.max()
    stats[f"eval_gen/train/word_lengths_min"] = word_lengths.min()

    stats[f"eval_gen/train/logprobs"] = logprobs.mean().item()
    stats[f"eval_gen/train/old_logprobs"] = old_logprobs.mean().item()
    stats[f"eval_gen/train/approxkl"] = (0.5 * ((logprobs - old_logprobs) ** 2).mean()).item()
    stats[f"eval_gen/train/policykl"] = (logprobs - old_logprobs).mean().item()
    stats[f"eval_gen/train/sequence_approxkl"] = (0.5 * ((logprobs - old_logprobs) ** 2)).sum(-1).mean().item()
    stats[f"eval_gen/train/sequence_policykl"] = (logprobs - old_logprobs).sum(-1).mean().item()

    # log table of completions
    table_rows = list(r for r in zip(*[all_to_log[col] for col in columns_to_log_eval]))
    stats[f"eval_gen/train/epoch_{epoch}_generations"] = wandb.Table(columns=[*columns_to_log_eval], rows=table_rows)


def __eval_metrics(trainer, dataloader, tokenizer, split: str = "train"):
    log_main_process(trainer.accelerator, f"Computing evaluation metrics for split: {split}")
    preferred_logprobs = []
    reference_preferred_logprobs = []
    dispreferred_logprobs = []
    reference_dispreferred_logprobs = []
    output_lengths_w = []
    output_lengths_l = []

    for batch_index, pref_batch in tqdm(enumerate(dataloader), desc=f"Eval batches for split: {split}", total=len(dataloader)):
        input_ids_w, attention_mask_w, input_ids_l, attention_mask_l = __train_process_pref_batch(pref_batch, tokenizer, trainer)
        mask_w = trainer.get_prompt_mask(input_ids_w, attention_mask_w)
        mask_l = trainer.get_prompt_mask(input_ids_l, attention_mask_l)

        logprobs_w, old_logprobs_w, = __eval_process_input_ids(input_ids_w, attention_mask_w, trainer)
        logprobs_l, old_logprobs_l, = __eval_process_input_ids(input_ids_l, attention_mask_l, trainer)

        logprobs_w, old_logprobs_w = trainer.accelerator.gather_for_metrics((logprobs_w, old_logprobs_w))
        logprobs_l, old_logprobs_l = trainer.accelerator.gather_for_metrics((logprobs_l, old_logprobs_l))
        mask_w, mask_l = trainer.accelerator.gather_for_metrics((mask_w, mask_l))

        preferred_logprobs.append((logprobs_w * mask_w).sum(dim=-1).detach().cpu())
        reference_preferred_logprobs.append((old_logprobs_w * mask_w).sum(dim=-1).detach().cpu())
        dispreferred_logprobs.append((logprobs_l * mask_l).sum(dim=-1).detach().cpu())
        reference_dispreferred_logprobs.append((old_logprobs_l * mask_l).sum(dim=-1).detach().cpu())
        output_lengths_w.append(mask_w.sum(dim=-1).detach().cpu())
        output_lengths_l.append(mask_l.sum(dim=-1).detach().cpu())

    preferred_logprobs = torch.cat(preferred_logprobs)
    dispreferred_logprobs = torch.cat(dispreferred_logprobs)
    reference_preferred_logprobs = torch.cat(reference_preferred_logprobs)
    reference_dispreferred_logprobs = torch.cat(reference_dispreferred_logprobs)
    output_lengths_w = torch.cat(output_lengths_w)
    output_lengths_l = torch.cat(output_lengths_l)

    preferred_logprobs_mean = preferred_logprobs.mean()
    reference_preferred_logprobs_mean = reference_preferred_logprobs.mean()
    dispreferred_logprobs_mean = dispreferred_logprobs.mean()
    reference_dispreferred_logprobs_mean = reference_dispreferred_logprobs.mean()
    loss = trainer.compute_loss(preferred_logprobs, dispreferred_logprobs, reference_preferred_logprobs, reference_dispreferred_logprobs,
                                output_lengths_w, output_lengths_l)

    preferred_probs_mean = torch.exp(preferred_logprobs).mean()
    reference_preferred_probs_mean = torch.exp(reference_preferred_logprobs).mean()
    dispreferred_probs_mean = torch.exp(dispreferred_logprobs).mean()
    reference_dispreferred_probs_mean = torch.exp(reference_dispreferred_logprobs).mean()

    stats = {}
    stats[f"eval_metrics/{split}/loss"] = loss.item()
    stats[f"eval_metrics/{split}/preferred_logprobs_mean"] = preferred_logprobs_mean.item()
    stats[f"eval_metrics/{split}/reference_preferred_logprobs_mean"] = reference_preferred_logprobs_mean.item()
    stats[f"eval_metrics/{split}/preferred_logprobs_change_mean"] = (preferred_logprobs_mean - reference_preferred_logprobs_mean).item()
    stats[f"eval_metrics/{split}/dispreferred_logprobs_mean"] = dispreferred_logprobs_mean.item()
    stats[f"eval_metrics/{split}/reference_dispreferred_logprobs_mean"] = reference_dispreferred_logprobs_mean.item()
    stats[f"eval_metrics/{split}/dispreferred_logprobs_change_mean"] = (dispreferred_logprobs_mean - reference_dispreferred_logprobs_mean).item()
    stats[f"eval_metrics/{split}/pi_logratios_mean"] = (preferred_logprobs_mean - dispreferred_logprobs_mean).item()
    stats[f"eval_metrics/{split}/preference_classifier_accuracy"] = torch.mean((preferred_logprobs - dispreferred_logprobs > 0).float()).item()

    stats[f"eval_metrics/{split}/preferred_probs_mean"] = preferred_probs_mean.item()
    stats[f"eval_metrics/{split}/reference_preferred_probs_mean"] = reference_preferred_probs_mean.item()
    stats[f"eval_metrics/{split}/preferred_probs_change_mean"] = (preferred_probs_mean - reference_preferred_probs_mean).item()
    stats[f"eval_metrics/{split}/dispreferred_probs_mean"] = dispreferred_probs_mean.item()
    stats[f"eval_metrics/{split}/reference_dispreferred_probs_mean"] = reference_dispreferred_probs_mean.item()
    stats[f"eval_metrics/{split}/dispreferred_probs_change_mean"] = (dispreferred_probs_mean - reference_dispreferred_probs_mean).item()
    return stats


def __quantile_slice(array, quantile, num_elements):
    """
    Returns a slice of the array centered around the quantile and with the specified number of elements.

    Parameters:
    - array: The input array (list or numpy array).
    - quantile: The quantile (float between 0 and 1).
    - num_elements: The number of elements in the output slice (integer).
    """
    idx = int(np.round(quantile * (len(array) - 1)))

    half_window = num_elements // 2
    start = max(0, idx - half_window)
    end = start + num_elements

    if end > len(array):
        end = len(array)
        start = max(0, end - num_elements)

    return array[start:end]


def __subsample_dataset(dataset):
    if not FLAGS.preference_similarity_path:
        if FLAGS.preference_num_samples < 0:
            return dataset

        if FLAGS.random_seed_for_training_example_selection > 0:
            perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(FLAGS.random_seed_for_training_example_selection))
        else:
            perm = torch.randperm(len(dataset))

        num_samples = min(FLAGS.preference_num_samples, len(dataset))
        return dataset.select(perm[:num_samples])

    preference_similarity_info = torch.load(FLAGS.preference_similarity_path, map_location="cpu")
    sample_indices = preference_similarity_info["sample_indices"]
    dataset = dataset.select(sample_indices)

    pref_similarities = preference_similarity_info[FLAGS.pref_similarity_measure]

    sorting_order = torch.argsort(pref_similarities)
    quantile = FLAGS.use_samples_with_preference_similarity_quantile
    chosen_indices = __quantile_slice(sorting_order, quantile, FLAGS.preference_num_samples)
    return dataset.select(chosen_indices)


def __filter_queries_exceeding_max_length(dataset, tokenizer):
    def filter_fn(example):
        tokenized_query = tokenizer(example["query"], truncation=False, padding=False)
        return len(tokenized_query["input_ids"]) <= FLAGS.max_input_length

    return dataset.filter(filter_fn)


def __prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model, cache_dir=FLAGS.cache_dir)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": PADDING_TOKEN})

    tokenizer.add_special_tokens({"additional_special_tokens": [PROMPT_TOKEN, ASSISTANT_TOKEN]})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    return tokenizer


def __prepare_dataset(tokenizer, accelerator):
    if FLAGS.preference_dataset_path == "tatsu-lab/alpaca_farm":
        dataset = load_dataset(FLAGS.preference_dataset_path, "alpaca_human_preference", split="preference", cache_dir=FLAGS.cache_dir)

        def process_dataset(batch):
            new_batch = defaultdict(list)
            for inst, inp, out1, out2, pref in zip(batch["instruction"], batch["input"], batch["output_1"], batch["output_2"], batch["preference"]):
                if pref == 1:
                    selected = out1
                    rejected = out2
                else:
                    selected = out2
                    rejected = out1
                if inp:
                    text = f"{PROMPT_TOKEN}{inst}\n{inp}{ASSISTANT_TOKEN}"
                else:
                    text = f"{PROMPT_TOKEN}{inst}{ASSISTANT_TOKEN}"

                new_batch["query"].append(text)
                new_batch["text_w"].append(f"{text}{selected}{tokenizer.eos_token}")
                new_batch["text_l"].append(f"{text}{rejected}{tokenizer.eos_token}")
            return new_batch

        dataset = dataset.map(
            process_dataset,
            batched=True,
            num_proc=FLAGS.num_proc,
        )
    elif FLAGS.preference_dataset_path == "HuggingFaceH4/ultrafeedback_binarized":
        dataset = load_dataset(FLAGS.preference_dataset_path, split="train_prefs", cache_dir=FLAGS.cache_dir)

        def process_dataset(example):
            new_example = {}
            chosen, rejected = example["chosen"], example["rejected"]
            query = f"{PROMPT_TOKEN}{chosen[0]['content']}{ASSISTANT_TOKEN}"

            new_example["query"] = query
            new_example["text_w"] = f"{query}{chosen[1]['content']}{tokenizer.eos_token}"
            new_example["text_l"] = f"{query}{rejected[1]['content']}{tokenizer.eos_token}"
            return new_example

        dataset = dataset.map(process_dataset, batched=False)
    else:
        dataset = load_dataset(FLAGS.preference_dataset_path, cache_dir=FLAGS.cache_dir)

    def process_dataset(batch):
        new_batch = {}
        new_batch["query"] = batch["query"]
        new_batch["text_w"] = batch["text_w"]
        new_batch["text_l"] = batch["text_l"]
        new_batch["response_w"] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch["text_w"]]
        new_batch["response_l"] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch["text_l"]]

        return new_batch

    dataset = dataset.map(
        process_dataset,
        batched=True,
        num_proc=FLAGS.num_proc
    )

    keep_columns = ["query", "text_w", "text_l", "response_w", "response_l"]
    dataset = dataset.select_columns(keep_columns)
    dataset = __subsample_dataset(dataset)

    before_filter_dataset_size = len(dataset)
    dataset = __filter_queries_exceeding_max_length(dataset, tokenizer)
    log_main_process(accelerator, f"Filtered out {before_filter_dataset_size - len(dataset)} samples with prompts"
                                  f" exceeding max input length of {FLAGS.max_input_length}. Remaining samples: {len(dataset)}")
    return dataset


def log_main_process(accelerator, message: str, log_level=logging.INFO):
    if accelerator.is_main_process:
        logging_utils.log(level=log_level, msg=message)


def save_model(trainer, output_dir, checkpoint_dir, epoch_num, add_prefix=True):
    if add_prefix:
        checkpoint_dir = os.path.join(output_dir, checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

    if trainer.accelerator.is_main_process:
        trainer.accelerator.unwrap_model(trainer.model).save_pretrained(
            checkpoint_dir,
            save_function=trainer.accelerator.save,
            is_main_process=trainer.accelerator.is_main_process,
            state_dict=trainer.accelerator.get_state_dict(trainer.model),
        )
        trainer.tokenizer.save_pretrained(checkpoint_dir)
        log_main_process(trainer.accelerator, f"Checkpointing Epoch {epoch_num} -> {checkpoint_dir}")


@torch.no_grad()
def __compute_and_save_evaluation_metrics(epoch, trainer, output_dir, begin_eval_msg: str = ""):
    if begin_eval_msg:
        log_main_process(trainer.accelerator, begin_eval_msg)

    trainer.model.eval()
    eval_stats = {"epoch": epoch}
    eval_stats.update(__eval_metrics(trainer, trainer.dataloader, trainer.tokenizer, split="train"))

    if trainer.accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with jsonlines.open(os.path.join(output_dir, "eval_metrics.jsonl"), mode="a") as writer:
            writer.write(eval_stats)

    __eval_generations(epoch, trainer, trainer.tokenizer, eval_stats)
    trainer.log_stats(stats=eval_stats)

    log_main_process(trainer.accelerator, f"Finished evaluation at the end of epoch {epoch}")


def __train(trainer, output_dir, model_checkpoint_name):
    start_time = datetime.datetime.utcnow()
    log_main_process(trainer.accelerator, f"Output dir: {output_dir}")
    log_main_process(trainer.accelerator, f"Model name: {model_checkpoint_name}")
    log_main_process(trainer.accelerator, f"Number of train samples: {len(trainer.dataset)}")
    log_main_process(trainer.accelerator, f"Sample train example: {trainer.dataset[0]}")

    log_main_process(trainer.accelerator, f"Starting training for {FLAGS.num_train_epochs} epochs on {len(trainer.dataset)} examples")
    for epoch in tqdm(range(FLAGS.num_train_epochs), desc="Epochs"):
        train_batch_stats = []
        for batch_index, pref_batch in tqdm(enumerate(trainer.dataloader), desc="Train batches", total=len(trainer.dataloader)):
            with trainer.accelerator.accumulate(trainer.model):
                input_ids_w, attention_mask_w, input_ids_l, attention_mask_l = __train_process_pref_batch(pref_batch, trainer.tokenizer, trainer)
                train_stats = trainer.step(input_ids_w, attention_mask_w, input_ids_l, attention_mask_l)

            train_stats["epoch"] = epoch
            trainer.log_stats(stats=train_stats)
            train_batch_stats.append(train_stats)

        agg_train_epoch_stats = __aggregate_training_batch_stats(train_batch_stats)
        agg_train_epoch_stats["epoch"] = epoch
        if trainer.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            with jsonlines.open(os.path.join(output_dir, "train_metrics.jsonl"), mode="a") as writer:
                writer.write(agg_train_epoch_stats)

        if (epoch + 1) % FLAGS.eval_every_epochs == 0:
            __compute_and_save_evaluation_metrics(epoch, trainer, output_dir,
                                                  begin_eval_msg=f"Computing evaluation metrics at the end of epoch {epoch}")

        if (epoch + 1) % FLAGS.save_every_epochs == 0 and FLAGS.save_every_epochs > 0:
            save_model(trainer, output_dir, model_checkpoint_name + f"_epoch_{epoch}", epoch)

    if FLAGS.save_at_end:
        save_model(trainer, output_dir, model_checkpoint_name + f"_epoch_{FLAGS.num_train_epochs - 1}", FLAGS.num_train_epochs - 1)

    trainer.end_training()
    end_time = datetime.datetime.utcnow()
    log_main_process(trainer.accelerator, f"Finished run, time took: {end_time - start_time}")


def main(_):
    unique_str = datetime.datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S") + "-" + str(np.random.randint(10000))
    run_name = f"{FLAGS.run_name}_quantile_{FLAGS.use_samples_with_preference_similarity_quantile}_{unique_str}"
    output_dir = f"{FLAGS.output_dir}/{FLAGS.wandb_project}/{run_name}"
    model_checkpoint_name = f"{FLAGS.run_name}_checkpoint"

    set_seed(FLAGS.seed)
    try:
        config = DPOConfig(
            model_name=FLAGS.model,
            gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size_per_gpu_proc,
            tracker_project_name=FLAGS.wandb_project,
            objective=FLAGS.objective,
            kl_coeff=FLAGS.kl_coeff,
            sft_coeff=FLAGS.sft_coeff,
            project_kwargs={
                "project_dir": output_dir,
            },
            tracker_kwargs={
                "wandb": {
                    "name": run_name,
                    "id": unique_str
                }
            },
            log_with="wandb",
            seed=FLAGS.seed,
        )

        accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )

        logging_utils.init_console_logging()
        if accelerator.is_main_process:
            logging_utils.init_file_logging(log_file_name_prefix="log", output_dir=output_dir)
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(FLAGS.flag_values_dict(), f, indent=2)

        # Prepares dataset, but does not tokenize it (tokenization is done per batch online)
        tokenizer = __prepare_tokenizer()
        pref_dataset = __prepare_dataset(tokenizer, accelerator)

        device_map = "auto" if FLAGS.model_parallel_without_accelerate else None
        model = AutoModelForCausalLM.from_pretrained(
            FLAGS.model,
            cache_dir=FLAGS.cache_dir,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.resize_token_embeddings(len(tokenizer))

        trainer = DPOTrainer(
            accelerator=accelerator,
            config=config,
            model=model,
            dataset=pref_dataset,
            tokenizer=tokenizer,
            optimizer_name=FLAGS.optimizer_name,
            additional_config_kwargs=FLAGS.flag_values_dict(),
            assistant_token=ASSISTANT_TOKEN
        )

        __train(trainer, output_dir, model_checkpoint_name)
    except Exception:
        logging_utils.exception("Exception occurred during training.")
        raise


if __name__ == "__main__":
    app.run(main)
