import typing
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from accelerate import Accelerator
from torch import nn
from transformers import PreTrainedTokenizerBase

from common_dpo.trainers.dpo_config import DPOConfig
from common_dpo.trainers.utils import (
    logprobs_from_logits,
    flatten_dict,
    is_torch_greater_2_0,
    create_reference_model
)

PreTrainedModelWrapper = typing.Union[nn.Module, nn.DataParallel]


class DPOTrainer:

    def __init__(
            self,
            accelerator: Accelerator = None,
            config: DPOConfig = None,
            model: PreTrainedModelWrapper = None,
            dataset: torch.utils.data.Dataset = None,
            ref_model: Optional[PreTrainedModelWrapper] = None,
            tokenizer: PreTrainedTokenizerBase = None,
            assistant_token: str = None,
            optimizer_name: str = None,
            num_shared_layers: Optional[int] = None,
            lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            prompt_end_positions_func: Optional[callable] = None,
            additional_config_kwargs: Optional[dict] = None
    ):
        self.accelerator = accelerator
        self.config = config
        if self.config.objective not in ["dpo", "ipo", "cross_entropy"]:
            raise ValueError(f"Objective {self.config.objective} not supported")

        self.model = model
        if ref_model is None:
            self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
        else:
            self.ref_model = ref_model

        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        if self.is_encoder_decoder:
            raise ValueError("Reinforce does not support encoder-decoder models.")

        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=None,
            shuffle=True
        )
        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        current_config = dict(trl_reinforce_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict()
        current_config.update(flatten_dict(additional_config_kwargs or {}))

        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=current_config,
            init_kwargs=config.tracker_kwargs,
        )
        self.tokenizer = tokenizer
        self.assistant_token = assistant_token
        self.prompt_end_positions_func = prompt_end_positions_func

        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        elif optimizer_name == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            lr_scheduler_class = (
                torch.optim.lr_scheduler._LRScheduler
                if not is_torch_greater_2_0()
                else torch.optim.lr_scheduler.LRScheduler
            )

            if not isinstance(self.lr_scheduler, lr_scheduler_class):
                raise ValueError(
                    "lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.LRScheduler (for torch >= 2.0)"
                )

        # Safety checkers for DS integration
        is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )

        (
            self.model,
            self.dataloader,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader,
            self.optimizer,
            self.lr_scheduler,
        )
        if is_deepspeed_used:
            # Quantized models are already set on the correct device
            if not self.is_peft_model and not (
                    getattr(self.ref_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(self.ref_model.pretrained_model, "is_loaded_in_4bit", False)
            ):
                self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)

        # In a distributed setup, only logging needs to be performed on the main process
        # check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"

        # init the current step
        self.current_step = 0

    def __process_input_ids(self, input_ids, attention_mask, compute_ref_logprobs: bool = True):
        input_data = {"input_ids": input_ids, "attention_mask": attention_mask}
        logits = self.model(**input_data).logits
        logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

        if not compute_ref_logprobs:
            return logprobs

        with torch.no_grad():
            old_logits = self.ref_model(**input_data).logits
            old_logprobs = logprobs_from_logits(old_logits[:, :-1, :], input_ids[:, 1:])

        return logprobs, old_logprobs

    def get_prompt_mask(self, input_ids, attention_mask):
        if not self.assistant_token and self.prompt_end_positions_func is None:
            return attention_mask[:, :-1]

        # mask all pad and prompt tokens
        if self.prompt_end_positions_func is None:
            assistant_token_id = self.tokenizer.vocab[self.assistant_token]
            prompt_end_positions = (input_ids == assistant_token_id).nonzero(as_tuple=True)[1]
        else:
            prompt_end_positions = self.prompt_end_positions_func(input_ids, self.tokenizer)

        row_indices = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).repeat(input_ids.size(0), 1)
        mask = (row_indices >= prompt_end_positions.unsqueeze(1))[:, :-1].to(torch.int64)
        return mask

    def __compute_dpo_loss(self, seq_logprobs_w, seq_logprobs_l, seq_old_logprobs_w, seq_old_logprobs_l):
        pi_logratio = seq_logprobs_w - seq_logprobs_l
        ref_logratio = seq_old_logprobs_w - seq_old_logprobs_l
        return -F.logsigmoid(self.config.kl_coeff * (pi_logratio - ref_logratio)).mean()

    def __compute_ipo_loss(self, seq_logprobs_w, seq_logprobs_l, seq_old_logprobs_w, seq_old_logprobs_l):
        pi_logratio = seq_logprobs_w - seq_logprobs_l
        ref_logratio = seq_old_logprobs_w - seq_old_logprobs_l
        return ((pi_logratio - ref_logratio - 1 / (2 * self.config.kl_coeff)) ** 2).mean()

    def compute_loss(self, seq_logprobs_w, seq_logprobs_l=None, seq_old_logprobs_w=None, seq_old_logprobs_l=None,
                     output_lengths_w=None, output_lengths_l=None):
        if self.config.objective == "cross_entropy":
            loss = - seq_logprobs_w.mean()
        elif self.config.objective == "dpo":
            loss = self.__compute_dpo_loss(seq_logprobs_w, seq_logprobs_l, seq_old_logprobs_w, seq_old_logprobs_l)
        elif self.config.objective == "ipo":
            loss = self.__compute_ipo_loss(seq_logprobs_w, seq_logprobs_l, seq_old_logprobs_w, seq_old_logprobs_l)
        else:
            raise ValueError(f"Objective {self.config.objective} not supported")

        if self.config.sft_coeff > 0 and self.config.objective != "cross_entropy":
            loss += -self.config.sft_coeff * seq_logprobs_w.mean()

        return loss

    def __perform_cross_entropy_step(self, input_ids_w, attention_mask_w, return_stats):
        logprobs_w = self.__process_input_ids(input_ids_w, attention_mask_w, compute_ref_logprobs=False)
        mask_w = self.get_prompt_mask(input_ids_w, attention_mask_w)

        seq_logprobs_w = (logprobs_w * mask_w).sum(dim=-1)
        loss = - seq_logprobs_w.mean()

        if return_stats:
            stats = {
                "train/loss": loss.detach().cpu().item(),
                "train/preferred_logprobs_mean": seq_logprobs_w.mean().detach().cpu().item()
            }
            return loss, stats
        else:
            return loss

    def __perform_dpo_or_ipo_step(self, input_ids_w, attention_mask_w, input_ids_l, attention_mask_l, return_stats):
        logprobs_w, old_logprobs_w = self.__process_input_ids(input_ids_w, attention_mask_w)
        logprobs_l, old_logprobs_l = self.__process_input_ids(input_ids_l, attention_mask_l)

        mask_w = self.get_prompt_mask(input_ids_w, attention_mask_w)
        mask_l = self.get_prompt_mask(input_ids_l, attention_mask_l)

        seq_logprobs_w = (logprobs_w * mask_w).sum(dim=-1)
        seq_logprobs_l = (logprobs_l * mask_l).sum(dim=-1)
        seq_old_logprobs_w = (old_logprobs_w * mask_w).sum(dim=-1)
        seq_old_logprobs_l = (old_logprobs_l * mask_l).sum(dim=-1)

        if self.config.objective == "dpo":
            loss = self.__compute_dpo_loss(seq_logprobs_w, seq_logprobs_l, seq_old_logprobs_w, seq_old_logprobs_l)
        else:
            loss = self.__compute_ipo_loss(seq_logprobs_w, seq_logprobs_l, seq_old_logprobs_w, seq_old_logprobs_l)

        if self.config.sft_coeff > 0:
            loss += -self.config.sft_coeff * seq_logprobs_w.mean()

        pi_logratio = seq_logprobs_w - seq_logprobs_l
        log_ratio_diff = pi_logratio - seq_old_logprobs_w - seq_old_logprobs_l

        rewards_chosen = self.config.kl_coeff * (seq_logprobs_w - seq_old_logprobs_w).detach()
        rewards_rejected = self.config.kl_coeff * (seq_logprobs_l - seq_old_logprobs_l).detach()
        reward_margin = rewards_chosen - rewards_rejected
        if return_stats:
            stats = {
                "train/loss": loss.detach().cpu().item(),
                "train/rewards_mean": 0.5 * (rewards_chosen.mean() + rewards_rejected.mean()).cpu().item(),
                "train/rewards_chosen_mean": rewards_chosen.mean().cpu().item(),
                "train/rewards_rejected_mean": rewards_rejected.mean().cpu().item(),
                "train/reward_margin_mean": reward_margin.mean().cpu().item(),
                "train/preferred_logprobs_mean": seq_logprobs_w.mean().detach().cpu().item(),
                "train/dispreferred_logprobs_mean": seq_logprobs_l.mean().detach().cpu().item(),
                "train/reference_preferred_logprobs_mean": seq_old_logprobs_w.mean().detach().cpu().item(),
                "train/reference_dispreferred_logprobs_mean": seq_old_logprobs_l.mean().detach().cpu().item(),
                "train/pi_logratios_mean": pi_logratio.mean().detach().cpu().item(),
                "train/pi_reference_log_ratio_diff_mean": log_ratio_diff.mean().detach().cpu().item(),
                "train/preference_classifier_accuracy": torch.mean((pi_logratio > 0).float()).cpu().item()
            }
            return loss, stats
        else:
            return loss

    def _step(
            self,
            input_ids_w: torch.LongTensor,
            attention_mask_w: torch.Tensor,
            input_ids_l: torch.LongTensor,
            attention_mask_l: torch.Tensor,
            return_stats: bool = False
    ):
        if self.config.objective == "cross_entropy":
            return self.__perform_cross_entropy_step(input_ids_w, attention_mask_w, return_stats)
        elif self.config.objective in ["dpo", "ipo"]:
            return self.__perform_dpo_or_ipo_step(input_ids_w, attention_mask_w, input_ids_l, attention_mask_l, return_stats)
        else:
            raise ValueError(f"Objective {self.config.objective} not supported")

    def step(
            self,
            input_ids_w: torch.LongTensor,
            attention_mask_w: torch.Tensor,
            input_ids_l: torch.LongTensor,
            attention_mask_l: torch.Tensor,
    ):
        self.model.train()

        loss, stats = self._step(
            input_ids_w=input_ids_w,
            attention_mask_w=attention_mask_w,
            input_ids_l=input_ids_l,
            attention_mask_l=attention_mask_l,
            return_stats=True,
        )

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.current_step += 1

        return stats

    def log_stats(self, stats: dict):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
        """
        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            logs = {}
            logs.update(stats)

            # manually cast in fp32 for bf16 torch tensors
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            self.accelerator.log(logs, step=self.current_step)

    def end_training(self):
        self.accelerator.end_training()
