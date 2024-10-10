import torch

from ..trainer import Trainer
from ...evaluation.evaluators.evaluator import VoidEvaluator


class SupervisedTrainer(Trainer):
    """
    Trainer for regular supervised task of predicting y given x (classification or regression).
    """

    def __init__(self, model, optimizer, loss_fn, train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(), callback=None,
                 gradient_accumulation: int = -1, device=torch.device("cpu")):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.loss_fn = loss_fn
        self.gradient_accumulation = gradient_accumulation

    def batch_update(self, batch_num, batch, total_num_batches):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)

        loss = self.loss_fn(y_pred, y)
        if self.gradient_accumulation > 0:
            loss = loss / self.gradient_accumulation

        loss.backward()

        do_accumulated_grad_update = (batch_num + 1) % self.gradient_accumulation == 0 or batch_num == total_num_batches - 1
        if self.gradient_accumulation <= 0 or do_accumulated_grad_update:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "loss": loss.item(),
            "y_pred": y_pred.detach(),
            "y": y
        }
