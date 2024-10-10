from typing import List

import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from accelerate import Accelerator

from common.data.modules import DataModule


class PersonaSingleOutputTokenDataModuleAccelerate(DataModule):
    def __init__(self, path: str, model: nn.Module, tokenizer, output_tokens_matching_yes: List[str], output_tokens_matching_no: List[str],
                 num_train_samples: int, accelerator: Accelerator, answer_matching_behavior_to_use: str = "", batch_size: int = -1,
                 pin_memory: bool = False, random_seed: int = -1, ):
        self.path = path
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.output_tokens_matching_yes = output_tokens_matching_yes
        self.output_tokens_matching_no = output_tokens_matching_no

        self.__verify_tokens(self.output_tokens_matching_yes)
        self.__verify_tokens(self.output_tokens_matching_no)

        self.num_train_samples = num_train_samples
        self.answer_matching_behavior_to_use = answer_matching_behavior_to_use
        self.batch_size = batch_size
        self.pin_memory = pin_memory

        self.random_seed = random_seed
        self.accelerator = accelerator

    def __verify_tokens(self, tokens: List[str]):
        vocab = self.tokenizer.get_vocab()
        for token in tokens:
            if token not in vocab:
                raise ValueError(f"Token '{token}' is not in the tokenizer's vocabulary.")

    def setup(self):
        self.dataset = datasets.load_dataset("json", data_files=self.path, split="train")

        if self.answer_matching_behavior_to_use:
            self.dataset = self.dataset.filter(lambda example: example["answer_matching_behavior"].strip().lower()
                                                               == self.answer_matching_behavior_to_use.strip().lower())

        if self.random_seed > 0:
            perm = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(self.random_seed))
        else:
            perm = torch.randperm(len(self.dataset))

        if self.num_train_samples > 0 and self.num_train_samples < len(self.dataset):
            self.dataset = self.dataset.select(perm[:self.num_train_samples])

        self.dataset = self.dataset.map(self.__prepare_example_for_preference_based_finetuning, batched=True)
        self.tokenized_dataset = self.__tokenize_dataset(self.dataset)

        self.model.eval()
        self.tokenized_dataset = self.tokenized_dataset.map(self.__add_reference_log_probs, batched=True, batch_size=self.batch_size)

    def __prepare_example_for_preference_based_finetuning(self, example: dict) -> dict:
        if not self.tokenizer.chat_template:
            new_example = {"input": [question + "\n" for question in example["question"]]}
        else:
            chats = [[{"role": "user", "content": question}] for question in example["question"]]
            new_example = {"input": [self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]}

        new_example["preferred_output"] = [
            self.output_tokens_matching_yes[torch.randint(low=0, high=len(self.output_tokens_matching_yes), size=(1,))]
            if answer_matching_behavior.strip().lower() == "Yes".lower() else
            self.output_tokens_matching_no[torch.randint(low=0, high=len(self.output_tokens_matching_no), size=(1,))]
            for answer_matching_behavior in example["answer_matching_behavior"]
        ]
        new_example["dispreferred_output"] = [
            self.output_tokens_matching_no[torch.randint(low=0, high=len(self.output_tokens_matching_no), size=(1,))]
            if answer_matching_behavior.strip().lower() == "Yes".lower() else
            self.output_tokens_matching_yes[torch.randint(low=0, high=len(self.output_tokens_matching_yes), size=(1,))]
            for answer_matching_behavior in example["answer_matching_behavior"]
        ]

        return new_example

    def __tokenize_example(self, example: dict) -> dict:
        tokenized_input = self.tokenizer(example["input"],
                                         add_special_tokens=False if self.tokenizer.chat_template else True,
                                         padding=True,
                                         return_tensors="pt")
        tokenized_preferred_output_ids = self.tokenizer.convert_tokens_to_ids(example["preferred_output"])
        tokenized_dispreferred_output_ids = self.tokenizer.convert_tokens_to_ids(example["dispreferred_output"])
        return {
            "input_ids": tokenized_input["input_ids"],
            "attention_mask": tokenized_input["attention_mask"],
            "preferred_output_ids": tokenized_preferred_output_ids,
            "dispreferred_output_ids": tokenized_dispreferred_output_ids
        }

    def __tokenize_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
        dataset = dataset.map(self.__tokenize_example, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "preferred_output_ids", "dispreferred_output_ids"])
        dataset = dataset.select_columns(["input_ids", "attention_mask", "preferred_output_ids", "dispreferred_output_ids"])
        return dataset

    def __add_reference_log_probs(self, example: dict) -> dict:
        input_ids = example["input_ids"].to(self.accelerator.device)
        attention_mask = example["attention_mask"].to(self.accelerator.device)
        preferred_output_ids = example["preferred_output_ids"].to(self.accelerator.device)
        dispreferred_output_ids = example["dispreferred_output_ids"].to(self.accelerator.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "logits"):
                all_logits = outputs.logits
                output_logits = all_logits[:, -1, :]
            else:
                output_logits = outputs

            output_logprobs = F.log_softmax(output_logits, dim=1)

        preferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids]
        dispreferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), dispreferred_output_ids]
        example.update({"ref_preferred_logprobs": preferred_logprobs, "ref_dispreferred_logprobs": dispreferred_logprobs})
        return example

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        shuffle = batch_size < len(self.tokenized_dataset)
        dataloader = torch.utils.data.DataLoader(self.tokenized_dataset, batch_size=batch_size, shuffle=shuffle)

        if self.accelerator:
            return self.accelerator.prepare(dataloader)

        return dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        dataloader = torch.utils.data.DataLoader(self.tokenized_dataset, batch_size=batch_size, shuffle=False)

        if self.accelerator:
            return self.accelerator.prepare(dataloader)

        return dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        dataloader = torch.utils.data.DataLoader(self.tokenized_dataset, batch_size=batch_size, shuffle=False)

        if self.accelerator:
            return self.accelerator.prepare(dataloader)

        return dataloader
