import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from .metric import ScalarMetric, AveragedMetric


class MSELoss(AveragedMetric):
    """
    MSE loss metric.
    """

    def __init__(self):
        super().__init__()

    def _calc_metric(self, y_pred, y):
        """
        Calculates the mean square error loss.
        :param y_pred: predictions.
        :param y: true values.
        :return: (Mean square error loss, num samples in input)
        """
        loss = F.mse_loss(y_pred, y)
        return loss.item(), len(y)


class ClippedMSELoss(AveragedMetric):
    """
    MSE loss metric, with per sample loss clipped at some threshold.
    """

    def __init__(self, max_sample_loss: float = -1):
        """
        :param max_sample_loss: the square loss of a sample is clipped at this value (i.e. taken to be the minimum between it and the loss). If < 0,
        will not clip the loss.
        """
        super().__init__()
        self.max_sample_loss = max_sample_loss

    def _calc_metric(self, y_pred, y):
        if self.max_sample_loss < 0:
            return F.mse_loss(y_pred, y).item(), len(y)

        clipped_square_losses = torch.clamp((y_pred - y) ** 2, max=self.max_sample_loss)
        return clipped_square_losses.mean().item(), len(y)


class L1Loss(AveragedMetric):
    """
    L1 loss metric.
    """

    def __init__(self):
        super().__init__()

    def _calc_metric(self, y_pred, y):
        """
        Calculates the L1 loss.
        :param y_pred: predictions.
        :param y: true values.
        :return: (L1 loss, num samples in input)
        """
        loss = F.l1_loss(y_pred, y)
        return loss.item(), len(y)


class SmoothL1Loss(AveragedMetric):
    """
    Smooth L1 loss (Huber loss) metric.
    """

    def __init__(self, beta: float = 1.0):
        """
        :param beta: specifies the threshold at which to change between L1 and L2 loss. This value defaults to 1.0.
        """
        super().__init__()
        self.beta = beta

    def _calc_metric(self, y_pred, y):
        """
        Calculates the smooth L1 loss.
        :param y_pred: predictions.
        :param y: true values.
        :return: (Smooth L1 loss, num samples in input)
        """
        loss = F.smooth_l1_loss(y_pred, y, beta=self.beta)
        return loss.item(), len(y)


class Correlation(ScalarMetric):
    """
    Correlation metric. Supports Pearson and Spearman correlations.
    """

    def __init__(self, correlation_type="pearson"):
        self.correlation_type = correlation_type
        self.corr_func = self.__get_correlation_func(correlation_type)
        self.predictions = []
        self.true_values = []

    @staticmethod
    def __get_correlation_func(correlation_type):
        if correlation_type == "pearson":
            return pearsonr
        elif correlation_type == "spearman":
            return spearmanr
        else:
            raise ValueError(f"Unsupported correlation type {correlation_type}. Supported types are: 'pearson', 'spearman'.")

    def __call__(self, y_pred, y):
        y_pred = y_pred.squeeze().detach().cpu().numpy()
        y = y.squeeze().detach().cpu().numpy()

        corr = self.corr_func(y_pred, y)[0]
        self.predictions.append(y_pred)
        self.true_values.append(y)
        return corr.item()

    def current_value(self):
        y_pred = torch.cat(self.predictions, dim=0)
        y = torch.cat(self.true_values, dim=0)
        return self.corr_func(y_pred, y)[0].item()

    def has_epoch_metric_to_update(self):
        return len(self.predictions) > 0

    def reset_current_epoch_values(self):
        self.predictions = []
        self.true_values = []


class ReconstructionError(AveragedMetric):
    """Reconstruction error metric."""

    def __init__(self, normalized=False):
        super().__init__()
        self.normalized = normalized

    def _calc_metric(self, tensor: torch.Tensor, target_tensor: torch.Tensor):
        reconstruction_error = compute_reconstruction_error(tensor, target_tensor, self.normalized)
        return reconstruction_error, 1


def compute_reconstruction_error(tensor: torch.Tensor, target_tensor: torch.Tensor, normalize: bool = False):
    """
    Computes the reconstruction error which is the Frobenius distance between the tensors, possibly normalized by the norm of the target tensor.
    :param tensor: torch tensor.
    :param target_tensor: torch tensor.
    :param normalize: if True, normalizes the reconstruction error with the target tensor norm.
    """
    reconstruction_error = torch.norm(tensor - target_tensor, p="fro").item()
    if normalize:
        target_norm = torch.norm(target_tensor, p="fro").item()
        if target_norm != 0:
            reconstruction_error /= target_norm

    return reconstruction_error


class PredictionScaleMean(AveragedMetric):
    """
    For 1D outputs computes mean of predictions in absolute value. If outputs are multi-dimensional, computes mean L2 norm of predictions.
    """

    def _calc_metric(self, y_pred, y):
        """
        :param y_pred: predictions.
        :param y: true values.
        :return: (metric value, num samples in input)
        """
        flattened_predictions = y_pred.view(y_pred.shape[0], -1)
        multiple_outputs = flattened_predictions.shape[1] > 1

        predictions_scale_mean = y_pred.norm(dim=1).mean().item() if multiple_outputs else y_pred.abs().mean().item()
        return predictions_scale_mean, y_pred.shape[0]


class PredictionScaleMax(ScalarMetric):
    """
    For 1D outputs computes max of predictions in absolute value. If outputs are multi-dimensional, computes max L2 norm of predictions.
    """

    def __init__(self):
        self.current_epoch_max = float("-inf")
        self.computed_in_this_epoch = False

    def __call__(self, y_pred, y):
        """
        :param y_pred: predictions.
        :param y: true values.
        :return: (metric value, 1)
        """
        flattened_predictions = y_pred.view(y_pred.shape[0], -1)
        multiple_outputs = flattened_predictions.shape[1] > 1

        predictions_scale_max = y_pred.norm(dim=1).max().item() if multiple_outputs else y_pred.abs().max().item()

        self.current_epoch_max = max(predictions_scale_max, self.current_epoch_max)
        self.computed_in_this_epoch = True

        return predictions_scale_max

    def current_value(self):
        return self.current_epoch_max

    def has_epoch_metric_to_update(self):
        return self.computed_in_this_epoch

    def reset_current_epoch_values(self):
        self.current_epoch_max = float("-inf")
        self.computed_in_this_epoch = False
