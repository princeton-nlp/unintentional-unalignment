import torch
import torch.nn.functional as F

from .metric import AveragedMetric


class TripletMarginLoss(AveragedMetric):
    """
    Triplet margin loss metric. Calculates the PyTorch TripletMarginLoss on the given triplets.
    """

    def __init__(self, margin=1.0, reduction="mean"):
        """
        :param margin: margin for the triplet loss.
        :param reduction: reduction method param as supported by PyTorch TripletMarginLoss. Currently supports 'mean', 'sum' and 'none'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def _calc_metric(self, query, positive, negative):
        """
        :param query: query tensors batch.
        :param positive: positive tensors batch.
        :param negative: negative tensors batch.
        :return: (triplet margin loss, num samples in input)
        """
        loss = F.triplet_margin_loss(query, positive, negative, margin=self.margin, reduction=self.reduction)
        return loss.item(), len(query)


class TripletAccuracy(AveragedMetric):
    """
    Triplet accuracy metric. A correct triplet is one where the positive example is closer to the query than the negative example.
    """

    def __init__(self, margin=0):
        """
        :param margin: margin by which the negative distance should be greated than the positive one to be considered correct.
        """
        super().__init__()
        self.margin = margin

    def _calc_metric(self, query, positive, negative):
        """
        :param query: query tensors batch.
        :param positive: positive tensors batch.
        :param negative: negative tensors batch.
        :return: (triplet accuracy, num samples in input)
        """
        positive_distances = torch.norm(query - positive, dim=1)
        negative_distances = torch.norm(query - negative, dim=1)
        return (positive_distances + self.margin < negative_distances).sum().item() / len(query), len(query)
