import torch

from ..trainer import Trainer
from ...evaluation.evaluators.evaluator import VoidEvaluator


class MultitaskSupervisedTrainer(Trainer):
    """
    Trainer for multitask supervised tasks learning of predicting multiple outputs given x (classification or regression).
    """

    def __init__(self, model, optimizer, by_task_loss_functions, by_task_loss_weights=None,
                 train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(),
                 callback=None, device=torch.device("cpu")):
        """
        :param model: model that outputs a dictionary with name and output for each task.
        :param optimizer: optimizer.
        :param by_task_loss_functions: dictionary of loss functions, names should match outputs of model.
        :param by_task_loss_weights: dictionary of weights for the corresponding loss functions.
        :param train_evaluator: train phase evaluator.
        :param val_evaluator: validation phase evaluator.
        :param callback: callback for the training process.
        :param device: device to run on.
        """
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.by_task_loss_functions = by_task_loss_functions
        self.by_task_loss_weights = by_task_loss_weights if by_task_loss_weights is not None else {}

    def batch_update(self, batch_num, batch, total_num_batches):
        self.optimizer.zero_grad()

        x, by_task_y = batch
        x = x.to(self.device)
        by_task_y = {task_name: y.to(self.device) for task_name, y in by_task_y.items()}

        by_task_y_pred = self.model(x)
        by_task_loss = self.__calculate_by_task_losses(by_task_y_pred, by_task_y)

        total_loss = self.__calculate_total_loss(by_task_loss)
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "by_task_loss": {name: loss.item() for name, loss in by_task_loss.items()},
            "by_task_y_pred": {task_name: task_y_pred.detach() for task_name, task_y_pred in by_task_y_pred.items()},
            "by_task_y": by_task_y
        }

    def __calculate_by_task_losses(self, by_task_y_preds, by_task_y):
        by_task_losses = {}
        for name, loss_fn in self.by_task_loss_functions.items():
            y_pred = by_task_y_preds[name]
            y = by_task_y[name]

            loss = loss_fn(y_pred, y)
            by_task_losses[name] = loss

        return by_task_losses

    def __calculate_total_loss(self, by_task_losses):
        losses = [self.by_task_loss_weights.get(name, 1) * loss for name, loss in by_task_losses.items()]
        return sum(losses)
