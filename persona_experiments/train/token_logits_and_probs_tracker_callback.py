import logging
import math
from typing import List

import torch
from tabulate import tabulate

from common.train.callbacks import Callback
from common.train.trainer import Trainer


class TokenLogitsAndProbsTrackerCallback(Callback):

    def __init__(self, tokenizer, logger: logging.Logger, num_inputs_to_log_logit_change_for: int = 1, top_tokens_to_log: int = 10,
                 epoch_log_interval: int = 1, log_after_first_epoch: bool = True, track_finegrained_token_metrics: bool = False):
        self.tokenizer = tokenizer
        self.logger = logger
        self.num_inputs_to_log_logit_change_for = num_inputs_to_log_logit_change_for
        self.top_tokens_to_log = top_tokens_to_log
        self.epoch_log_interval = epoch_log_interval
        self.log_after_first_epoch = log_after_first_epoch
        self.track_finegrained_token_metrics = track_finegrained_token_metrics

        # Initial quantities
        self.initial_logprobs = None
        self.initial_preferred_logprobs = None
        self.initial_dispreferred_logprobs = None
        self.initial_preferred_logprobs_mean = None
        self.initial_hidden_representations = None

        # Counters
        self.prev_epoch_logged_due_to_preferred_prob_decrease = -math.inf
        self.num_steps_preferred_prob_decreased = 0
        self.num_steps_train_loss_increase_when_preferred_prob_decreased = 0

        # Previous step quantities
        self.prev_epoch_logprobs = None
        self.prev_epoch_loss = None
        self.prev_epoch_token_unembeddings = None
        self.prev_epoch_hidden_representations = None

        # Aggregate quantities
        self.min_pref_logprob = None
        self.min_pref_logprob_smaller_than_init = False

        # Per example/step quantities
        self.per_step_per_example_is_pref_logprob_smaller_than_init = []
        self.per_step_per_example_did_preferred_prob_decrease = []
        self.per_step_did_train_loss_decrease = []
        self.per_step_per_example_preferred_token_logprob_change = []
        self.per_step_per_example_dispreferred_token_logprob_change = []

        self.until_step_per_example_preferred_token_logprob_increase_rank = []
        self.until_step_per_example_preferred_token_prob_increase_rank = []
        self.until_step_per_example_top_logprob_increase_token_ids = []
        self.until_step_per_example_top_logprob_increase_values = []
        self.until_step_per_example_top_prob_increase_token_ids = []
        self.until_step_per_example_top_prob_increase_values = []

    @torch.no_grad()
    def on_train_batch_end(self, trainer: Trainer, batch_num: int, batch_output, metric_values):
        output_logprobs = batch_output["output logprobs"]
        input_ids = batch_output["input ids"]
        preferred_output_ids = batch_output["preferred output ids"]
        dispreferred_output_ids = batch_output["dispreferred output ids"]
        train_loss = batch_output["train loss"]
        unembedding_weights = batch_output["unembedding weights"]
        hidden_representations = batch_output["hidden representations"]

        if self.initial_logprobs is None:
            self.initial_logprobs = output_logprobs
            self.initial_preferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids]
            self.initial_dispreferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), dispreferred_output_ids]
            self.initial_preferred_logprobs_mean = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids].mean().item()
            self.initial_hidden_representations = hidden_representations
        else:
            logged = self.__log_if_preferred_token_prob_in_curr_step_decreased(trainer.epoch, output_logprobs, input_ids, preferred_output_ids,
                                                                               dispreferred_output_ids, train_loss, hidden_representations)

            should_log_token_logit_and_prob_stats = (self.epoch_log_interval > 0 and
                                                     ((trainer.epoch + 1) % self.epoch_log_interval == 0 or trainer.epoch == 1))
            if should_log_token_logit_and_prob_stats and not logged:
                self.__log_token_logit_and_prob_stats_for_examples(trainer.epoch, output_logprobs, input_ids,
                                                                   preferred_output_ids, dispreferred_output_ids, hidden_representations)
            if self.track_finegrained_token_metrics:
                self.__update_per_example_and_step_quantities(output_logprobs, preferred_output_ids, dispreferred_output_ids, train_loss)

        self.__update_prev_epoch_and_aggregate_quantities(output_logprobs, preferred_output_ids, train_loss,
                                                          unembedding_weights, hidden_representations)

    def __update_prev_epoch_and_aggregate_quantities(self, output_logprobs, preferred_output_ids, train_loss, unembedding_weights,
                                                     hidden_representations):
        self.prev_epoch_loss = train_loss
        self.prev_epoch_logprobs = output_logprobs

        curr_pref_log_prob = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids].mean().item()
        self.min_pref_logprob = min(self.min_pref_logprob, curr_pref_log_prob) if self.min_pref_logprob is not None else curr_pref_log_prob
        self.min_pref_logprob_smaller_than_init = (self.min_pref_logprob_smaller_than_init or
                                                   self.min_pref_logprob < self.initial_preferred_logprobs_mean)

        self.prev_epoch_token_unembeddings = unembedding_weights
        self.prev_epoch_hidden_representations = hidden_representations

    def __update_per_example_and_step_quantities(self, output_logprobs, preferred_output_ids, dispreferred_output_ids, train_loss):
        preferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids]
        dispreferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), dispreferred_output_ids]

        initial_preferred_logprobs = self.initial_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids]
        self.per_step_per_example_is_pref_logprob_smaller_than_init.append((preferred_logprobs < initial_preferred_logprobs).cpu())
        initial_dispreferred_logprobs = self.initial_logprobs[torch.arange(output_logprobs.size(0)), dispreferred_output_ids]

        prev_epoch_preferred_logprobs = self.prev_epoch_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids]
        self.per_step_per_example_did_preferred_prob_decrease.append((preferred_logprobs < prev_epoch_preferred_logprobs).cpu())
        self.per_step_per_example_preferred_token_logprob_change.append((preferred_logprobs - initial_preferred_logprobs).cpu())
        self.per_step_per_example_dispreferred_token_logprob_change.append((dispreferred_logprobs - initial_dispreferred_logprobs).cpu())
        self.per_step_did_train_loss_decrease.append(train_loss <= self.prev_epoch_loss)

        # Until step prob/log prob increase quantities
        overall_logprobs_change = output_logprobs - self.initial_logprobs
        overall_sorted_logprobs_change_indices = torch.argsort(overall_logprobs_change, dim=1, descending=True)
        overall_logprob_change_ranks_of_preferred_token = torch.where(overall_sorted_logprobs_change_indices == preferred_output_ids.view(-1, 1))[1]
        self.until_step_per_example_preferred_token_logprob_increase_rank.append(overall_logprob_change_ranks_of_preferred_token.cpu())

        overall_probs_change = torch.exp(output_logprobs) - torch.exp(self.initial_logprobs)
        overall_sorted_probs_change_indices = torch.argsort(overall_probs_change, dim=1, descending=True)
        overall_prob_change_ranks_of_preferred_token = torch.where(overall_sorted_probs_change_indices == preferred_output_ids.view(-1, 1))[1]
        self.until_step_per_example_preferred_token_prob_increase_rank.append(overall_prob_change_ranks_of_preferred_token.cpu())

        self.until_step_per_example_top_logprob_increase_token_ids.append(overall_sorted_logprobs_change_indices[:, :self.top_tokens_to_log].cpu())
        self.until_step_per_example_top_logprob_increase_values.append(torch.gather(overall_logprobs_change, dim=1,
                                                                                    index=overall_sorted_logprobs_change_indices[:,
                                                                                          :self.top_tokens_to_log]).cpu())
        self.until_step_per_example_top_prob_increase_token_ids.append(overall_sorted_probs_change_indices[:, :self.top_tokens_to_log].cpu())
        self.until_step_per_example_top_prob_increase_values.append(torch.gather(overall_probs_change, dim=1,
                                                                                 index=overall_sorted_probs_change_indices[:,
                                                                                       :self.top_tokens_to_log]).cpu())

    def __log_token_logit_and_prob_stats_for_examples(self, epoch: int, output_logprobs: torch.Tensor, input_ids: torch.Tensor,
                                                      preferred_output_ids: torch.Tensor, dispreferred_output_ids: torch.Tensor,
                                                      hidden_representations: torch.Tensor):
        for i in range(self.num_inputs_to_log_logit_change_for):
            self.__log_token_logit_and_prob_table(epoch, i, torch.exp(output_logprobs[i]), output_logprobs[i], input_ids[i],
                                                  preferred_output_ids[i], dispreferred_output_ids[i], hidden_representations[i])

    def __log_if_preferred_token_prob_in_curr_step_decreased(self, epoch: int, output_logprobs: torch.Tensor, input_ids: torch.Tensor,
                                                             preferred_output_ids: torch.Tensor, dispreferred_output_ids: torch.Tensor,
                                                             train_loss: float, hidden_representations: torch.Tensor):
        mean_preferred_log_probs = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids].mean()
        mean_prev_epoch_preferred_logprobs = self.prev_epoch_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids].mean()

        prob_decreased = (mean_preferred_log_probs - mean_prev_epoch_preferred_logprobs) < 0
        if not prob_decreased or self.prev_epoch_logged_due_to_preferred_prob_decrease + self.epoch_log_interval > epoch:
            return False

        self.prev_epoch_logged_due_to_preferred_prob_decrease = epoch
        self.num_steps_preferred_prob_decreased += 1
        train_loss_increased = train_loss > self.prev_epoch_loss
        if train_loss_increased:
            self.num_steps_train_loss_increase_when_preferred_prob_decreased += 1

        if self.epoch_log_interval <= 0:
            return False

        self.logger.info(f"\n**********************************************************\n"
                         f"Epoch: {epoch}: Mean preferred token log probability decreased from the previous step: "
                         f"{mean_preferred_log_probs.item():.6f} - {mean_prev_epoch_preferred_logprobs.item():.6f} "
                         f"(diff: {(mean_preferred_log_probs - mean_prev_epoch_preferred_logprobs).item():.6f}, "
                         f"init: {self.initial_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids].mean().item():.6f})\n"
                         f"Train loss after step {train_loss:.6f} and before step {self.prev_epoch_loss:.6f} (increased? {train_loss_increased})\n"
                         f"Total number of steps in which preferred token probability decreased thus far: {self.num_steps_preferred_prob_decreased}\n"
                         f"**********************************************************")

        self.__log_token_logit_and_prob_stats_for_examples(epoch, output_logprobs, input_ids,
                                                           preferred_output_ids, dispreferred_output_ids, hidden_representations)
        return True

    def __log_token_logit_and_prob_table(self, epoch: int, example_index: int, output_probs: torch.Tensor,
                                         output_logprobs: torch.Tensor, input_ids: torch.Tensor,
                                         preferred_output_id: torch.Tensor, dispreferred_output_id: torch.Tensor,
                                         hidden_representation: torch.Tensor):
        input = self.tokenizer.decode(input_ids)
        preferred_output = self.tokenizer.decode(preferred_output_id)
        dispreferred_output = self.tokenizer.decode(dispreferred_output_id)
        table_title = (f"Top {self.top_tokens_to_log} token logit and probability statistics for input: '{input}' ,"
                       f" preferred output: '{preferred_output}' , dispreferred output: '{dispreferred_output}'")

        initial_probs = torch.exp(self.initial_logprobs[example_index])
        prev_epoch_probs = torch.exp(self.prev_epoch_logprobs[example_index])
        curr_step_logprobs_change = output_logprobs - self.prev_epoch_logprobs[example_index]
        curr_step_probs_change = output_probs - prev_epoch_probs
        overall_logprobs_change = output_logprobs - self.initial_logprobs[example_index]
        overall_probs_change = output_probs - initial_probs

        table_row_values = []
        self.__populate_row_values_with_curr_step_token_prob_changes(table_row_values, example_index, output_probs, output_logprobs, initial_probs,
                                                                     prev_epoch_probs, curr_step_probs_change, curr_step_logprobs_change)
        self.__populate_row_values_with_overall_token_prob_changes(table_row_values, example_index, output_probs, output_logprobs, initial_probs,
                                                                   overall_probs_change, overall_logprobs_change)
        additional_info = self.__populate_row_values_with_curr_step_representation_metrics(table_row_values, example_index, curr_step_logprobs_change,
                                                                                           curr_step_probs_change, preferred_output_id,
                                                                                           dispreferred_output_id, hidden_representation)

        self.__log_table(epoch, table_title, row_values=table_row_values, additional_info=additional_info)

    def __populate_row_values_with_curr_step_token_prob_changes(self, row_values, example_index, output_probs, output_logprobs, initial_probs,
                                                                prev_epoch_probs, curr_step_probs_change, curr_step_logprobs_change):
        curr_step_top_logprob_increase_token_ids = torch.topk(curr_step_logprobs_change, k=self.top_tokens_to_log, largest=True).indices
        curr_step_top_logprob_increase_tokens = self.tokenizer.convert_ids_to_tokens(curr_step_top_logprob_increase_token_ids.view(-1, 1))
        curr_step_top_logprob_increase_decoded_tokens = self.tokenizer.batch_decode(curr_step_top_logprob_increase_token_ids.view(-1, 1))
        curr_step_top_logprob_increase_cell_strings = [(f"{token} (dec: {repr(decoded_token)}): {new_logprob:.6f} - {prev_logprob:.6f} "
                                                        f"(diff: {change:.6f}, init: {init_logprob:.6f})")
                                                       for token, decoded_token, new_logprob, prev_logprob, init_logprob, change in
                                                       zip(curr_step_top_logprob_increase_tokens,
                                                           curr_step_top_logprob_increase_decoded_tokens,
                                                           output_logprobs[curr_step_top_logprob_increase_token_ids],
                                                           self.prev_epoch_logprobs[example_index][curr_step_top_logprob_increase_token_ids],
                                                           self.initial_logprobs[example_index][curr_step_top_logprob_increase_token_ids],
                                                           curr_step_logprobs_change[curr_step_top_logprob_increase_token_ids])]

        row_values.append(["Curr Step Top Logprob Increase"] + curr_step_top_logprob_increase_cell_strings)

        curr_step_top_logprob_decrease_token_ids = torch.topk(curr_step_logprobs_change, k=self.top_tokens_to_log, largest=False).indices
        curr_step_top_logprob_decrease_tokens = self.tokenizer.convert_ids_to_tokens(curr_step_top_logprob_decrease_token_ids.view(-1, 1))
        curr_step_top_logprob_decrease_decoded_tokens = self.tokenizer.batch_decode(curr_step_top_logprob_decrease_token_ids.view(-1, 1))
        curr_step_top_logprob_decrease_cell_strings = [(f"{token} (dec: {repr(decoded_token)}): {new_logprob:.6f} - {prev_logprob:.6f} "
                                                        f"(diff: {change:.6f}, init: {init_logprob:.6f})")
                                                       for token, decoded_token, new_logprob, prev_logprob, init_logprob, change in
                                                       zip(curr_step_top_logprob_decrease_tokens,
                                                           curr_step_top_logprob_decrease_decoded_tokens,
                                                           output_logprobs[curr_step_top_logprob_decrease_token_ids],
                                                           self.prev_epoch_logprobs[example_index][curr_step_top_logprob_decrease_token_ids],
                                                           self.initial_logprobs[example_index][curr_step_top_logprob_decrease_token_ids],
                                                           curr_step_logprobs_change[curr_step_top_logprob_decrease_token_ids])]

        row_values.append(["Curr Step Top Logprob Decrease"] + curr_step_top_logprob_decrease_cell_strings)

        curr_step_top_prob_increase_token_ids = torch.topk(curr_step_probs_change, k=self.top_tokens_to_log, largest=True).indices
        curr_step_top_prob_increase_tokens = self.tokenizer.convert_ids_to_tokens(curr_step_top_prob_increase_token_ids.view(-1, 1))
        curr_step_top_prob_increase_decoded_tokens = self.tokenizer.batch_decode(curr_step_top_prob_increase_token_ids.view(-1, 1))
        curr_step_top_prob_increase_cell_strings = [(f"{token} (dec: {repr(decoded_token)}): {new_prob:.6f} - {prev_prob:.6f} (diff: {change:.6f}, "
                                                     f"init: {init_prob:.6f})")
                                                    for token, decoded_token, new_prob, prev_prob, init_prob, change in
                                                    zip(curr_step_top_prob_increase_tokens,
                                                        curr_step_top_prob_increase_decoded_tokens,
                                                        output_probs[curr_step_top_prob_increase_token_ids],
                                                        prev_epoch_probs[curr_step_top_prob_increase_token_ids],
                                                        initial_probs[curr_step_top_prob_increase_token_ids],
                                                        curr_step_probs_change[curr_step_top_prob_increase_token_ids])]

        row_values.append(["Curr Step Top Prob Increase"] + curr_step_top_prob_increase_cell_strings)

        curr_step_top_prob_decrease_token_ids = torch.topk(curr_step_probs_change, k=self.top_tokens_to_log, largest=False).indices
        curr_step_top_prob_decrease_tokens = self.tokenizer.convert_ids_to_tokens(curr_step_top_prob_decrease_token_ids.view(-1, 1))
        curr_step_top_prob_decrease_decoded_tokens = self.tokenizer.batch_decode(curr_step_top_prob_decrease_token_ids.view(-1, 1))
        curr_step_top_prob_decrease_cell_strings = [
            f"{token} (dec: {repr(decoded_token)}): {new_prob:.6f} - {prev_prob:.6f} (diff: {change:.6f}, init: {init_prob:.6f})"
            for token, decoded_token, new_prob, prev_prob, init_prob, change in
            zip(curr_step_top_prob_decrease_tokens,
                curr_step_top_prob_decrease_decoded_tokens,
                output_probs[curr_step_top_prob_decrease_token_ids],
                prev_epoch_probs[curr_step_top_prob_decrease_token_ids],
                initial_probs[curr_step_top_prob_decrease_token_ids],
                curr_step_probs_change[curr_step_top_prob_decrease_token_ids])]

        row_values.append(["Curr Step Top Prob Decrease"] + curr_step_top_prob_decrease_cell_strings)

    def __populate_row_values_with_overall_token_prob_changes(self, row_values, example_index, output_probs, output_logprobs, initial_probs,
                                                              overall_probs_change, overall_logprobs_change):
        overall_top_logprob_increase_token_ids = torch.topk(overall_logprobs_change, k=self.top_tokens_to_log, largest=True).indices
        overall_top_logprob_increase_tokens = self.tokenizer.convert_ids_to_tokens(overall_top_logprob_increase_token_ids.view(-1, 1))
        overall_top_logprob_increase_decoded_tokens = self.tokenizer.batch_decode(overall_top_logprob_increase_token_ids.view(-1, 1))
        overall_top_logprob_increase_cell_strings = [
            f"{token} (dec: {repr(decoded_token)}): {new_logprob:.6f} - {init_logprob:.6f} (diff: {change:.6f})"
            for token, decoded_token, new_logprob, init_logprob, change in
            zip(overall_top_logprob_increase_tokens,
                overall_top_logprob_increase_decoded_tokens,
                output_logprobs[overall_top_logprob_increase_token_ids],
                self.initial_logprobs[example_index][overall_top_logprob_increase_token_ids],
                overall_logprobs_change[overall_top_logprob_increase_token_ids])]

        row_values.append(["Overall Top Logprob Increase"] + overall_top_logprob_increase_cell_strings)

        overall_top_logprob_decrease_token_ids = torch.topk(overall_logprobs_change, k=self.top_tokens_to_log, largest=False).indices
        overall_top_logprob_decrease_tokens = self.tokenizer.convert_ids_to_tokens(overall_top_logprob_decrease_token_ids.view(-1, 1))
        overall_top_logprob_decrease_decoded_tokens = self.tokenizer.batch_decode(overall_top_logprob_decrease_token_ids.view(-1, 1))
        overall_top_logprob_decrease_cell_strings = [
            f"{token} (dec: {repr(decoded_token)}): {new_logprob:.6f} - {init_logprob:.6f} (diff: {change:.6f})"
            for token, decoded_token, new_logprob, init_logprob, change in
            zip(overall_top_logprob_decrease_tokens,
                overall_top_logprob_decrease_decoded_tokens,
                output_logprobs[overall_top_logprob_decrease_token_ids],
                self.initial_logprobs[example_index][overall_top_logprob_decrease_token_ids],
                overall_logprobs_change[overall_top_logprob_decrease_token_ids])]

        row_values.append(["Overall Top Logprob Decrease"] + overall_top_logprob_decrease_cell_strings)

        overall_top_prob_increase_token_ids = torch.topk(overall_probs_change, k=self.top_tokens_to_log, largest=True).indices
        overall_top_prob_increase_tokens = self.tokenizer.convert_ids_to_tokens(overall_top_prob_increase_token_ids.view(-1, 1))
        overall_top_prob_increase_decoded_tokens = self.tokenizer.batch_decode(overall_top_prob_increase_token_ids.view(-1, 1))
        overall_top_prob_increase_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {new_prob:.6f} - {init_prob:.6f} (diff: {change:.6f})"
                                                  for token, decoded_token, new_prob, init_prob, change in
                                                  zip(overall_top_prob_increase_tokens,
                                                      overall_top_prob_increase_decoded_tokens,
                                                      output_probs[overall_top_prob_increase_token_ids],
                                                      initial_probs[overall_top_prob_increase_token_ids],
                                                      overall_probs_change[overall_top_prob_increase_token_ids])]

        row_values.append(["Overall Top Prob Increase"] + overall_top_prob_increase_cell_strings)

        overall_top_prob_decrease_token_ids = torch.topk(overall_probs_change, k=self.top_tokens_to_log, largest=False).indices
        overall_top_prob_decrease_tokens = self.tokenizer.convert_ids_to_tokens(overall_top_prob_decrease_token_ids.view(-1, 1))
        overall_top_prob_decrease_decoded_tokens = self.tokenizer.batch_decode(overall_top_prob_decrease_token_ids.view(-1, 1))
        overall_top_prob_decrease_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {new_prob:.6f} - {init_prob:.6f} (diff: {change:.6f})"
                                                  for token, decoded_token, new_prob, init_prob, change in
                                                  zip(overall_top_prob_decrease_tokens,
                                                      overall_top_prob_decrease_decoded_tokens,
                                                      output_probs[overall_top_prob_decrease_token_ids],
                                                      initial_probs[overall_top_prob_decrease_token_ids],
                                                      overall_probs_change[overall_top_prob_decrease_token_ids])]

        row_values.append(["Overall Top Prob Decrease"] + overall_top_prob_decrease_cell_strings)

        curr_top_prob_token_ids = torch.topk(output_probs, k=self.top_tokens_to_log, largest=True).indices
        curr_top_prob_tokens = self.tokenizer.convert_ids_to_tokens(curr_top_prob_token_ids.view(-1, 1))
        curr_top_prob_decoded_tokens = self.tokenizer.batch_decode(curr_top_prob_token_ids.view(-1, 1))
        curr_top_prob_tokens_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {curr_prob:.6f} (initial: {init_prob:.6f})"
                                             for token, decoded_token, curr_prob, init_prob in
                                             zip(curr_top_prob_tokens,
                                                 curr_top_prob_decoded_tokens,
                                                 output_probs[curr_top_prob_token_ids],
                                                 initial_probs[curr_top_prob_token_ids])]
        curr_top_logprob_tokens_cell_strings = [(f"{token} (dec: {repr(decoded_token)}): {torch.log(curr_prob):.6f}"
                                                 f" (initial: {torch.log(init_prob):.6f})")
                                                for token, decoded_token, curr_prob, init_prob in
                                                zip(curr_top_prob_tokens,
                                                    curr_top_prob_decoded_tokens,
                                                    output_probs[curr_top_prob_token_ids],
                                                    initial_probs[curr_top_prob_token_ids])]

        row_values.append(["Curr Top Prob"] + curr_top_prob_tokens_cell_strings)
        row_values.append(["Curr Top Logprob"] + curr_top_logprob_tokens_cell_strings)

        initial_top_prob_token_ids = torch.topk(initial_probs, k=self.top_tokens_to_log, largest=True).indices
        initial_top_prob_tokens = self.tokenizer.convert_ids_to_tokens(initial_top_prob_token_ids.view(-1, 1))
        initial_top_prob_decoded_tokens = self.tokenizer.batch_decode(initial_top_prob_token_ids.view(-1, 1))
        initial_top_prob_tokens_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {init_prob:.6f}"
                                                for token, decoded_token, init_prob in
                                                zip(initial_top_prob_tokens,
                                                    initial_top_prob_decoded_tokens,
                                                    initial_probs[initial_top_prob_token_ids])]
        initial_top_logprob_tokens_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {torch.log(init_prob):.6f}"
                                                   for token, decoded_token, init_prob in
                                                   zip(initial_top_prob_tokens,
                                                       initial_top_prob_decoded_tokens,
                                                       initial_probs[initial_top_prob_token_ids])]

        row_values.append(["Initial Top Prob"] + initial_top_prob_tokens_cell_strings)
        row_values.append(["Initial Top Logprob"] + initial_top_logprob_tokens_cell_strings)

    def __populate_row_values_with_curr_step_representation_metrics(self, row_values, example_index, curr_step_logprobs_change,
                                                                    curr_step_probs_change, preferred_output_id, dispreferred_output_id,
                                                                    hidden_representation):
        preferred_token_embedding = self.prev_epoch_token_unembeddings[preferred_output_id]
        dispreferred_token_embedding = self.prev_epoch_token_unembeddings[dispreferred_output_id]
        unembedding_inner_prods = torch.matmul(self.prev_epoch_token_unembeddings, preferred_token_embedding - dispreferred_token_embedding)

        top_logprob_increase_token_ids = torch.topk(curr_step_logprobs_change, k=self.top_tokens_to_log, largest=True).indices.cpu()
        top_logprob_tokens = self.tokenizer.convert_ids_to_tokens(top_logprob_increase_token_ids.view(-1, 1))
        top_logprob_decoded_tokens = self.tokenizer.batch_decode(top_logprob_increase_token_ids.view(-1, 1))
        top_logprob_inner_prod_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {inner_prod:.6f}"
                                               for token, decoded_token, inner_prod in
                                               zip(top_logprob_tokens,
                                                   top_logprob_decoded_tokens,
                                                   unembedding_inner_prods[top_logprob_increase_token_ids])]

        row_values.append(["<W_y , W_p - W_d> For Curr Step Top Logprob Increase"] + top_logprob_inner_prod_cell_strings)

        top_prob_increase_token_ids = torch.topk(curr_step_probs_change, k=self.top_tokens_to_log, largest=True).indices.cpu()
        top_prob_tokens = self.tokenizer.convert_ids_to_tokens(top_prob_increase_token_ids.view(-1, 1))
        top_prob_decoded_tokens = self.tokenizer.batch_decode(top_prob_increase_token_ids.view(-1, 1))
        top_prob_inner_prod_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {inner_prod:.6f}"
                                            for token, decoded_token, inner_prod in
                                            zip(top_prob_tokens,
                                                top_prob_decoded_tokens,
                                                unembedding_inner_prods[top_prob_increase_token_ids])]

        row_values.append(["<W_y , W_p - W_d> For Curr Step Top Prob Increase"] + top_prob_inner_prod_cell_strings)

        top_inner_prod_token_ids = torch.topk(unembedding_inner_prods, self.top_tokens_to_log, largest=True).indices.cpu()
        top_inner_prod_tokens = self.tokenizer.convert_ids_to_tokens(top_inner_prod_token_ids.view(-1, 1))
        top_inner_prod_decoded_tokens = self.tokenizer.batch_decode(top_inner_prod_token_ids.view(-1, 1))
        top_inner_prod_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {inner_prod:.6f}"
                                       for token, decoded_token, inner_prod in
                                       zip(top_inner_prod_tokens,
                                           top_inner_prod_decoded_tokens,
                                           unembedding_inner_prods[top_inner_prod_token_ids])]

        row_values.append(["Curr Step Top <W_y , W_p - W_d>"] + top_inner_prod_cell_strings)

        bottom_inner_prod_token_ids = torch.topk(unembedding_inner_prods, self.top_tokens_to_log, largest=False).indices.cpu()
        bottom_inner_prod_tokens = self.tokenizer.convert_ids_to_tokens(bottom_inner_prod_token_ids.view(-1, 1))
        bottom_inner_prod_decoded_tokens = self.tokenizer.batch_decode(bottom_inner_prod_token_ids.view(-1, 1))
        bottom_inner_prod_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {inner_prod:.6f}"
                                          for token, decoded_token, inner_prod in
                                          zip(bottom_inner_prod_tokens,
                                              bottom_inner_prod_decoded_tokens,
                                              unembedding_inner_prods[bottom_inner_prod_token_ids])]

        row_values.append(["Curr Step Bottom <W_y , W_p - W_d>"] + bottom_inner_prod_cell_strings)

        prev_epoch_hidden_representation = self.prev_epoch_hidden_representations[example_index]
        hidden_repr_inner_prod = torch.inner(prev_epoch_hidden_representation,
                                             (preferred_token_embedding - dispreferred_token_embedding).to(prev_epoch_hidden_representation.device))
        hidden_repr_sq_norm = (prev_epoch_hidden_representation ** 2).sum()
        hidden_repr_sq_dist_from_init = ((prev_epoch_hidden_representation - self.initial_hidden_representations[example_index]) ** 2).sum()
        hidden_repr_curr_step_sq_dist = ((prev_epoch_hidden_representation - hidden_representation) ** 2).sum()

        num_largest_inner_prod_and_logprob_change = self.__compute_intersection_size(top_logprob_increase_token_ids,
                                                                                     top_inner_prod_token_ids)
        num_largest_inner_prod_and_prob_change = self.__compute_intersection_size(top_prob_increase_token_ids,
                                                                                  top_inner_prod_token_ids)

        additional_info = (f"Additional representations info:\n"
                           f"<W_p , W_p - W_d>: {unembedding_inner_prods[preferred_output_id].item():.6f}  "
                           f", <W_d , W_p - W_d>: {unembedding_inner_prods[dispreferred_output_id].item():.6f}  "
                           f", <W_p ,W_d>: {torch.inner(preferred_token_embedding, dispreferred_token_embedding).item():.6f}  "
                           f", ||W_p||^2: {(preferred_token_embedding ** 2).sum().item():.6f}  "
                           f", ||W_d||^2 {(dispreferred_token_embedding ** 2).sum().item():.6f}  "
                           f", <h , W_p - W_d>: {hidden_repr_inner_prod.item():.6f}  "
                           f", <h , W_p>: {torch.inner(prev_epoch_hidden_representation, preferred_token_embedding.to(prev_epoch_hidden_representation.device)).item():.6f}  "
                           f", <h , W_d>: {torch.inner(prev_epoch_hidden_representation, dispreferred_token_embedding.to(prev_epoch_hidden_representation.device)).item():.6f}  "
                           f", ||h||^2: {hidden_repr_sq_norm.item()}  "
                           f", ||h - h_init||^2: {hidden_repr_sq_dist_from_init.item()}  "
                           f", ||h^+ - h||^2: {hidden_repr_curr_step_sq_dist.item()}\n"
                           f"Num tokens with highest <W_y , W_p - W_d> and largest curr step logprob change:"
                           f" {num_largest_inner_prod_and_logprob_change} / {self.top_tokens_to_log}\n"
                           f"Num tokens with highest <W_y , W_p - W_d> and largest curr step prob change:"
                           f" {num_largest_inner_prod_and_prob_change} / {self.top_tokens_to_log}")
        return additional_info

    def __compute_intersection_size(self, first_indices_tensor: torch.Tensor, second_indices_tensor: torch.Tensor):
        set1 = set(first_indices_tensor.tolist())
        set2 = set(second_indices_tensor.tolist())
        common_indices = set1.intersection(set2)
        return len(common_indices)

    def __log_table(self, epoch: int, title: str, row_values: List[List[str]], additional_info: str = None):
        log_str = (f"===========================================================================\n"
                   f"Epoch: {epoch}, {title}\n{tabulate(row_values, tablefmt='pretty')}\n"
                   f"===========================================================================")

        if additional_info:
            log_str += f"\n{additional_info}"

        self.logger.info(log_str)
