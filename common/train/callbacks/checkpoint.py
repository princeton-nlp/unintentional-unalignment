from __future__ import annotations

import os
from datetime import datetime
from typing import Callable, TYPE_CHECKING

import torch

from .callback import Callback

if TYPE_CHECKING:
    from ..trainer import Trainer


class Checkpoint(Callback):
    """
    Allows saving the trainer object (with all of its components) on epoch end. Will also persist trainer state at
    the end of the fitting process.

    Each save interval a checkpoint will be saved. If the number of checkpoints will be above the given number of allowed checkpoints then one of the
    existing checkpoints will be deleted before saving the new one. This means the newest checkpoint will always be saved. Deletion will be done by
    oldest or by worst if a score function is given.
    """

    DEFAULT_FOLDER_NAME = "checkpoints"
    CHECKPOINT_FILE_EXTENSION = "ckpt"

    def __init__(self, output_dir: str, folder_name: str = DEFAULT_FOLDER_NAME, create_dir: bool = True, save_interval: int = 1, n_saved: int = 1,
                 score_function: Callable[[Trainer], float] = None, score_name: str = "", largest: bool = True, save_as_state_dict: bool = True):
        """
        :param output_dir: directory for saved checkpoints folder.
        :param folder_name: folder name under output_dir to save checkpoints to.
        :param create_dir: flag whether to create the output directory if it doesn't exist.
        :param save_interval: per how may epochs should a checkpoint be created. Pass -1 to save only last checkpoint after finished fit.
        :param n_saved: max number of saved checkpoints. Will delete old/worst checkpoint.
        :param score_function: optional score function that receives a trainer object and returns a score. If none given oldest checkpoint will be
        deleted. If a score function is given then the worst will be deleted on exceeding n_saved.
        :param score_name: name of the score metric (will be used in saved file format).
        :param largest: flag whether the largest value of the score is best (false for worst).
        :param save_as_state_dict: flag whether to save the trainer as a state dict (recommended). If false it will be persisted using torch.save
        which can break on change of class location and fields.
        """
        self.output_dir = output_dir
        self.folder_name = folder_name
        self.checkpoints_dir = os.path.join(self.output_dir, self.folder_name)
        self.create_dir = create_dir

        self.save_interval = save_interval
        self.n_saved = n_saved
        if self.n_saved <= 0:
            raise ValueError("n_saved parameter should be > 0")

        self.score_function = score_function
        self.score_name = score_name
        self.escaped_score_name = score_name.lower().replace(" ", "_")
        self.largest = largest
        self.save_as_state_dict = save_as_state_dict

        self.existing_checkpoints = []
        self.existing_checkpoints_scores = []

    def on_fit_initialization(self, trainer):
        if self.create_dir and not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

    def on_epoch_end(self, trainer):
        if self.save_interval == -1:
            return

        if (trainer.epoch + 1) % self.save_interval == 0:
            self.__create_trainer_checkpoint(trainer)

    def on_fit_end(self, trainer, num_epochs_ran, fit_output):
        self.__create_trainer_checkpoint(trainer)

    def get_best_checkpoint_path(self):
        if not self.existing_checkpoints:
            return ""

        if self.score_function is None:
            return self.existing_checkpoints[-1]

        best_val = max(self.existing_checkpoints_scores) if self.largest else min(self.existing_checkpoints_scores)
        best_checkpoint_file_name = self.existing_checkpoints[self.existing_checkpoints_scores.index(best_val)]
        return os.path.join(self.checkpoints_dir, best_checkpoint_file_name)

    def __create_trainer_checkpoint(self, trainer):
        if len(self.existing_checkpoints) >= self.n_saved:
            self.__delete_checkpoint()

        if self.score_function is not None:
            score = self.score_function(trainer)
            checkpoint_file_name = self.__create_checkpoint_file_name(trainer.epoch, score)
            self.__save_trainer(trainer, checkpoint_file_name)
            self.existing_checkpoints_scores.append(score)
        else:
            checkpoint_file_name = self.__create_checkpoint_file_name(trainer.epoch)
            self.__save_trainer(trainer, checkpoint_file_name)

    def __delete_checkpoint(self):
        to_delete_index = self.__get_to_delete_index()

        file_name = self.existing_checkpoints[to_delete_index]
        to_remove_path = os.path.join(self.checkpoints_dir, file_name)
        if os.path.exists(to_remove_path):
            os.remove(to_remove_path)

        del self.existing_checkpoints[to_delete_index]
        if self.score_function is not None:
            del self.existing_checkpoints_scores[to_delete_index]

    def __create_checkpoint_file_name(self, epoch, score=None):
        now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
        score_str = f"_{self.escaped_score_name}_{score:.3f}" if score is not None else ""
        return f"{now_utc_str}{score_str}_epoch_{epoch}.{Checkpoint.CHECKPOINT_FILE_EXTENSION}"

    def __get_to_delete_index(self):
        if self.score_function is not None:
            return self.__get_worse_checkpoint_index()
        return 0

    def __get_worse_checkpoint_index(self):
        worst_val = min(self.existing_checkpoints_scores) if self.largest else max(self.existing_checkpoints_scores)
        return self.existing_checkpoints_scores.index(worst_val)

    def __save_trainer(self, trainer, checkpoint_file_name):
        trainer_checkpoint = trainer
        if self.save_as_state_dict:
            trainer_checkpoint = trainer.state_dict()

        torch.save(trainer_checkpoint, os.path.join(self.checkpoints_dir, checkpoint_file_name))
        self.existing_checkpoints.append(checkpoint_file_name)
