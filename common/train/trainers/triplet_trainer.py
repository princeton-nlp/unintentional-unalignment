import torch

from ..trainer import Trainer
from ...evaluation.evaluators.evaluator import VoidEvaluator


class TripletTrainer(Trainer):

    def __init__(self, model, optimizer, loss_fn, train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(), callback=None,
                 device=torch.device("cpu")):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.loss_fn = loss_fn

    def batch_update(self, batch_num, batch, total_num_batches):
        self.optimizer.zero_grad()

        query, positive, negative = batch
        query = query.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)

        query = self.model(query)
        positive = self.model(positive)
        negative = self.model(negative)

        loss = self.loss_fn(query, positive, negative)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "query": query.detach(),
            "positive": positive.detach(),
            "negative": negative.detach()
        }
