import torch

from ..trainer import Trainer
from ...evaluation.evaluators.evaluator import VoidEvaluator


class NegativeSamplingSoftmaxTrainer(Trainer):
    """
    Trainer for models with subsampling layer for classification (for example NegativeSamplingLinear. Subsampling requires the model to receive the
    correct labels in order to know the positive example. The model needs to have a subsample_forward method that is expected to receive a tuple
    of (x, y) and output a tuple of (y_pred, y) where the second y is the correct label for the subsampled predictions.
    """

    def __init__(self, model, optimizer, loss_fn, train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(),
                 callback=None, device=torch.device("cpu")):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.loss_fn = loss_fn

    def batch_update(self, batch_num, batch, total_num_batches):
        self.optimizer.zero_grad()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred, y = self.model.negative_sample_forward(x, y)

        loss = self.loss_fn(y_pred, y)

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "y_pred": y_pred.detach(),
            "y": y
        }
