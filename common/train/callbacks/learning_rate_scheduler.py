from .callback import *


class LearningRateScheduler(Callback):

    def __init__(self, lr_scheduler, logger=None):
        self.lr_scheduler = lr_scheduler
        self.logger = logger

    def on_epoch_train_end(self, trainer, metric_values):
        self.lr_scheduler.step()
        if self.logger is not None:
            learning_rates = [param_group["lr"] for param_group in self.lr_scheduler.optimizer.param_groups]
            self.logger.info(f"Learning rate scheduler step done at the end of epoch {self.lr_scheduler.last_epoch - 1} training step. "
                             f"Current learning rates are: {learning_rates}")

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
