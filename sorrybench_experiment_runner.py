import json
import logging
import os

os.environ["WANDB_SERVICE_WAIT"] = "10000"
os.environ["WANDB_INIT_TIMEOUT"] = "10000"
os.environ["WANDB_START_METHOD"] = "thread"
import warnings
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
from common_dpo.trainers.utils import logprobs_from_logits, set_seed
import common.utils.logging as logging_utils
from utils import sorry_bench_utils
from utils import script_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_path", "", help="the path to the dataset")
flags.DEFINE_string("model", "meta-llama/Meta-Llama-3-8B-Instruct", "the path to/name of the pretrained model")
flags.DEFINE_bool("use_gold_preferences", False, "whether to use the gold refusal and non-refusal preferences")
flags.DEFINE_integer("seed", -1, "the random seed to set at beginning of training")
flags.DEFINE_integer("random_seed_for_training_example_selection", -1, "random seed used for ensuring same training example selection")
flags.DEFINE_string("wandb_project", "sorrybench_dpo", "the wandb project name")
flags.DEFINE_string("run_name", "run", "the wandb run name")
flags.DEFINE_string("output_dir", "outputs", "the output directory")
flags.DEFINE_string("preference_similarity_path", "", "path to file with preference similarity data")
flags.DEFINE_float("use_samples_with_preference_similarity_quantile", 0,
                   "will use training samples whose preference similarity (in terms of hidden representations inner products or token overlap)"
                   " is centered around the given quantile")
flags.DEFINE_string("pref_similarity_measure", "ches_scores",
                    "the measure to use for choosing training samples according to preference similarity "
                    "(supports 'ches_scores', 'ln_ches_scores', 'minus_normalized_edit_distances', and 'last_hidden_embedding_inner_prods'")
flags.DEFINE_integer("preference_num_samples", -1, "the number of samples to use from the preference dataset")
flags.DEFINE_integer("batch_size_per_gpu_proc", 4, "the batch size per gpu process (with accelerate the true batch size will be "
                                                   "this value * num_gpus used, without accelerate this will be the batch size)")
flags.DEFINE_integer("gradient_accumulation_steps", 1, "the gradient accumulation steps")
flags.DEFINE_string("cache_dir", None, "the cache directory")
flags.DEFINE_string("optimizer_name", "rmsprop", "name of optimizer to use")
flags.DEFINE_float("learning_rate", 5e-6, "the learning rate")

flags.DEFINE_integer("num_train_epochs", 1, "the number of training epochs")
flags.DEFINE_integer("eval_every_epochs", 1, "how often to evaluate")
flags.DEFINE_integer("save_every_epochs", -1, "how often to save checkpoints")
flags.DEFINE_bool("save_at_end", False, "save checkpoint at the end of training")

flags.DEFINE_string("objective", "dpo", "which objective to use (supports 'dpo', 'ipo', and 'cross_entropy')")
flags.DEFINE_float("kl_coeff", 0.1, "the KL coefficient to use in the DPO and IPO objectives")
flags.DEFINE_float("sft_coeff", 0,
                   "if non-zero, will use an sft term in addition to dpo and ipo with this coefficient (e.g., as done in CPO, RPO, ORPO)")

flags.DEFINE_bool("model_parallel_without_accelerate", False,
                  "whether to use model parallelism (only use when running script without accelerate)")
flags.DEFINE_integer("refusal_eval_batch_size", 16, "batch size to use when evaluating refusals")

flags.DEFINE_bool("use_only_two_refusal_samples", False, "use only training examples with two refusal responses "
                                                         "(only relevant when not filtering based on preference similarity)")
flags.DEFINE_bool("use_only_two_non_refusal_samples", False, "use only training examples with two non-refusal responses "
                                                             "(only relevant when not filtering based on preference similarity)")
flags.DEFINE_bool("use_only_one_of_each_samples", False, "use only training examples with one refusal and another non-refusal responses "
                                                         "(only relevant when not filtering based on preference similarity)")


def __llama3_instruct_get_prompt_end_position_func(input_ids, tokenizer):
    end_header_token_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")

    mask = input_ids != end_header_token_id
    non_zero_indices = torch.arange(input_ids.size(1), device=mask.device).repeat(input_ids.size(0), 1)
    non_zero_indices[mask] = -1

    end_header_indices = non_zero_indices.max(dim=1).values
    return end_header_indices + 1  # After the end header token there is a '\n\n' token


def __gemma2b_instruct_get_prompt_end_position_func(input_ids, tokenizer):
    start_turn_token = tokenizer.convert_tokens_to_ids("<start_of_turn>")

    mask = input_ids != start_turn_token
    non_zero_indices = torch.arange(input_ids.size(1), device=mask.device).repeat(input_ids.size(0), 1)
    non_zero_indices[mask] = -1

    start_turn_indices = non_zero_indices.max(dim=1).values
    return start_turn_indices + 2  # After the start of turn token there are two tokens: 'model' and '\n'


PROMPT_END_POS_FUNCS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": __llama3_instruct_get_prompt_end_position_func,
    "google/gemma-2b-it": __gemma2b_instruct_get_prompt_end_position_func
}


def __aggregate_training_batch_stats(train_batch_stats):
    epoch_stats = {}
    metric_names = train_batch_stats[0].keys()
    for metric_name in metric_names:
        epoch_stats[metric_name] = np.mean([batch_stats[metric_name] for batch_stats in train_batch_stats])

    return epoch_stats


@torch.no_grad()
def __train_process_pref_batch(pref_batch, tokenizer, trainer, use_orig_rejected: bool = False):
    chosen = pref_batch["chosen"]
    rejected = pref_batch["rejected"] if not use_orig_rejected else pref_batch["orig_rejected"]
    queries = [[{"role": "user", "content": query}] for query in pref_batch["query"]]

    chosen_inputs = [tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True) + chosen_response + tokenizer.eos_token
                     for chosen_response, query in zip(chosen, queries)]
    rejected_inputs = [tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True) + reject_response + tokenizer.eos_token
                       for reject_response, query in zip(rejected, queries)]

    all_inputs = chosen_inputs + rejected_inputs

    tokenized_inputs = tokenizer(all_inputs, add_special_tokens=False, padding=True, return_tensors="pt")
    tokenized_input_ids = accelerate.utils.send_to_device(tokenized_inputs.input_ids, trainer.accelerator.device)
    attention_masks = accelerate.utils.send_to_device(tokenized_inputs.attention_mask, trainer.accelerator.device)

    input_ids_w = tokenized_input_ids[:len(chosen)]
    attention_mask_w = attention_masks[:len(chosen)]
    input_ids_l = tokenized_input_ids[len(chosen):]
    attention_mask_l = attention_masks[len(chosen):]

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
        if FLAGS.use_only_two_refusal_samples:
            dataset = dataset.filter(lambda example: example["chosen_score"] == 0 and example["rejected_score"] == 0)
        elif FLAGS.use_only_two_non_refusal_samples:
            dataset = dataset.filter(lambda example: example["chosen_score"] == 1 and example["rejected_score"] == 1)
        elif FLAGS.use_only_one_of_each_samples:
            dataset = dataset.filter(lambda example: example["chosen_score"] == 0 and example["rejected_score"] == 1)

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


def __create_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model, cache_dir=FLAGS.cache_dir)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    return tokenizer


def __prepare_datasets():
    train_dataset = load_dataset("json", data_files=os.path.join(FLAGS.dataset_path, "train.jsonl"), split="train")
    test_dataset = load_dataset("json", data_files=os.path.join(FLAGS.dataset_path, "test.jsonl"), split="train")

    if FLAGS.use_gold_preferences:
        def replace_preferences_with_gold(batch):
            batch["rejected"] = batch["gold_non_refusal"]
            batch["chosen"] = batch["gold_refusal"]
            return batch

        train_dataset = train_dataset.map(replace_preferences_with_gold, batched=True)
        test_dataset = test_dataset.map(replace_preferences_with_gold, batched=True)

    subsampled_train_dataset = __subsample_dataset(train_dataset)
    return train_dataset, subsampled_train_dataset, test_dataset


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
def __compute_and_save_evaluation_metrics(epoch, trainer, test_dataloader, output_dir, begin_eval_msg: str = ""):
    if begin_eval_msg:
        log_main_process(trainer.accelerator, begin_eval_msg)

    trainer.model.eval()
    eval_stats = {"epoch": epoch}
    eval_stats.update(__eval_metrics(trainer, trainer.dataloader, trainer.tokenizer, split="train"))
    eval_stats.update(__eval_metrics(trainer, test_dataloader, trainer.tokenizer, split="test"))

    trainer.log_stats(stats=eval_stats)
    if trainer.accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with jsonlines.open(os.path.join(output_dir, "eval_metrics.jsonl"), mode="a") as writer:
            writer.write(eval_stats)

    log_main_process(trainer.accelerator, f"Finished evaluation at the end of epoch {epoch}")


def __train(trainer, test_dataset, output_dir, model_checkpoint_name):
    log_main_process(trainer.accelerator, f"Output dir: {output_dir}")
    log_main_process(trainer.accelerator, f"Model name: {model_checkpoint_name}")
    log_main_process(trainer.accelerator, f"Number of train samples: {len(trainer.dataset)}")
    log_main_process(trainer.accelerator, f"Sample train example: {trainer.dataset[0]}")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=trainer.config.batch_size,
        collate_fn=None,
        shuffle=False
    )
    test_dataloader = trainer.accelerator.prepare(test_dataloader)

    __compute_and_save_evaluation_metrics(-1, trainer, test_dataloader, output_dir,
                                          begin_eval_msg=f"Computing evaluation metrics before start of training")

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
            __compute_and_save_evaluation_metrics(epoch, trainer, test_dataloader, output_dir,
                                                  begin_eval_msg=f"Computing evaluation metrics at the end of epoch {epoch}")

        if (epoch + 1) % FLAGS.save_every_epochs == 0 and FLAGS.save_every_epochs > 0:
            save_model(trainer, output_dir, model_checkpoint_name + f"_epoch_{epoch}", epoch)

    if FLAGS.save_at_end:
        save_model(trainer, output_dir, model_checkpoint_name + f"_epoch_{FLAGS.num_train_epochs - 1}", FLAGS.num_train_epochs - 1)

    trainer.end_training()


def __evaluate_refusal(accelerator, tokenizer, model, train_dataset, test_dataset, batch_size, output_dir):
    logging_utils.info("Evaluating refusal rates at the end of training")

    # Evaluate refusal only requires a single GPU
    train_queries = [example["query"] for example in train_dataset]
    logging_utils.info("Generating responses for the training set")
    train_responses = sorry_bench_utils.generate_responses_for_eval(tokenizer, model, queries=train_queries, batch_size=batch_size,
                                                                    logger=logging_utils, accelerator=accelerator)

    logging_utils.info("Generating responses for the test set")
    test_queries = [example["query"] for example in test_dataset]
    test_responses = sorry_bench_utils.generate_responses_for_eval(tokenizer, model, queries=test_queries, batch_size=batch_size,
                                                                   logger=logging_utils, accelerator=accelerator)

    # Delete model and free GPU memory for loading judge model
    del model
    accelerator.free_memory()
    torch.cuda.empty_cache()

    device = torch.device("cuda:0")
    judge_tokenizer, judge_model = script_utils.load_tokenizer_and_model(model_name=sorry_bench_utils.DEFAULT_JUDGE_MODEL_NAME,
                                                                         cache_dir=FLAGS.cache_dir,
                                                                         device=device)
    if not judge_tokenizer.pad_token:
        judge_tokenizer.pad_token = judge_tokenizer.unk_token if judge_tokenizer.unk_token else judge_tokenizer.eos_token

    judge_tokenizer.padding_side = "left"
    judge_tokenizer.truncation_side = "right"

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
    file_name = f"{FLAGS.model.split('/')[-1]}_refusal_eval.json"
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(refusal_metrics, f, indent=2)

    logging_utils.info(f"Saved refusal rate metrics at: {os.path.join(output_dir, file_name)}\n"
                       f"{refusal_metrics}")

    full_responses = {
        "train": [{"query": query, "response": response, "score": score} for query, response, score in
                  zip(train_queries, train_responses, train_scores)],
        "test": [{"query": query, "response": response, "score": score} for query, response, score in
                 zip(test_queries, test_responses, test_scores)]
    }
    responses_file_name = f"{FLAGS.model.split('/')[-1]}_full_responses.json"
    with open(os.path.join(output_dir, responses_file_name), "w") as f:
        json.dump(full_responses, f, indent=2)

    logging_utils.info(f"Saved full responses at: {os.path.join(output_dir, responses_file_name)}")


def main(_):
    unique_str = datetime.datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S") + "-" + str(np.random.randint(10000))
    run_name = f"{FLAGS.run_name}_{unique_str}"
    output_dir = f"{FLAGS.output_dir}/{FLAGS.wandb_project}/{run_name}"
    model_checkpoint_name = f"{FLAGS.run_name}_checkpoint"

    set_seed(FLAGS.seed)
    try:
        start_time = datetime.datetime.utcnow()

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
        tokenizer = __create_tokenizer()
        train_dataset, subsampled_train_dataset, test_dataset = __prepare_datasets()

        device_map = "auto" if FLAGS.model_parallel_without_accelerate else None
        model = AutoModelForCausalLM.from_pretrained(
            FLAGS.model,
            cache_dir=FLAGS.cache_dir,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        prompt_end_positions_func = PROMPT_END_POS_FUNCS[FLAGS.model] if FLAGS.model in PROMPT_END_POS_FUNCS else None
        trainer = DPOTrainer(
            accelerator=accelerator,
            config=config,
            model=model,
            dataset=subsampled_train_dataset,
            tokenizer=tokenizer,
            optimizer_name=FLAGS.optimizer_name,
            prompt_end_positions_func=prompt_end_positions_func,
            additional_config_kwargs=FLAGS.flag_values_dict()
        )

        __train(trainer, test_dataset, output_dir, model_checkpoint_name)
        if accelerator.is_main_process:
            __evaluate_refusal(accelerator, tokenizer, trainer.accelerator.unwrap_model(trainer.model), train_dataset, test_dataset,
                               batch_size=FLAGS.refusal_eval_batch_size, output_dir=output_dir)

        end_time = datetime.datetime.utcnow()
        log_main_process(trainer.accelerator, f"Finished run, time took: {end_time - start_time}")
    except Exception:
        logging_utils.exception("Exception occurred during training.")
        raise


if __name__ == "__main__":
    app.run(main)
