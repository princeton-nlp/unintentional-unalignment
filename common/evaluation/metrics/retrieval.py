from .metric import AveragedMetric


class ExactLabelHitCounter:
    """
    Callable for calculating number of hits based on exact match of labels.
    """

    def __call__(self, prediction, label):
        """
        :param prediction: tensor of shape (num_predictions,) with the predictions
        :param label: int label
        :return: Number of exact hits in the predictions for the label
        """
        return (prediction == label).sum().item()


class HitCounter:
    """
    Callable for calculating number of hits based on custom matching feature between the label item and the predictions.
    """

    def __init__(self, prediction_to_feature_mapper, label_to_feature_mapper=None):
        self.prediction_to_feature_mapper = prediction_to_feature_mapper

        if label_to_feature_mapper is not None:
            self.label_to_feature_mapper = label_to_feature_mapper
        else:
            self.label_to_feature_mapper = prediction_to_feature_mapper

    def __call__(self, prediction, label):
        """
        :param prediction: tensor of shape (num_predictions,) with the predictions
        :param label: int label
        :return: Number of hits according to feature of the predictions and label
        """
        hits = 0
        label_feature = self.label_to_feature_mapper[label]
        for predicted_label in prediction:
            if self.prediction_to_feature_mapper[predicted_label] == label_feature:
                hits += 1
        return hits


class TopKAccuracy(AveragedMetric):
    """
    Top K accuracy metric. Calculates the prediction accuracy where if in the top k predictions there is a match between the
    label and a prediction then it is considered a hit.
    """

    def __init__(self, hits_counter=ExactLabelHitCounter(), k=1, predictions_transform=lambda x: x, labels_transform=lambda x: x):
        """
        :param hits_counter: callable that counts the number of hits between a tensor of predictions and a label.
        :param k: k top results to consider.
        :param predictions_transform: transform to do on the predictions when metric is called.
        :param labels_transform: transform to do on the labels when metric is called.
        """
        super().__init__()
        self.hits_counter = hits_counter
        self.k = k
        self.predictions_transform = predictions_transform
        self.labels_transform = labels_transform

    def _calc_metric(self, predictions, labels):
        """
        :param predictions: tensor of shape (batch_size, num_predictions)
        :param labels: tensor of labels of size (batch_size,)
        :return: (Top k accuracy value, num samples in input)
        """
        hits = 0
        predictions = self.predictions_transform(predictions)
        labels = self.labels_transform(labels)
        predictions = predictions[:, : min(len(predictions[0]), self.k)]

        for prediction, label in zip(predictions, labels):
            if self.hits_counter(prediction, label) > 0:
                hits += 1

        return hits / len(labels), len(labels)


class PrecisionAtK(AveragedMetric):
    """
    Precision at k metric. Calculates the precision of the top 5 predictions.
    """

    def __init__(self, hits_counter=ExactLabelHitCounter(), k=1, predictions_transform=lambda x: x, labels_transform=lambda x: x):
        """
        :param hits_counter: callable that counts the number of hits between a tensor of predictions and a label.
        :param k: k top results to consider.
        :param predictions_transform: transform to do on the predictions when metric is called.
        :param labels_transform: transform to do on the labels when metric is called.
        """
        super().__init__()
        self.hits_counter = hits_counter
        self.k = k
        self.predictions_transform = predictions_transform
        self.labels_transform = labels_transform

    def _calc_metric(self, predictions, labels):
        """
        :param predictions: tensor of shape (batch_size, num_predictions)
        :param labels: tensor of labels of size (batch_size,)
        :return: (Precision@k value, num samples in input)
        """
        precision_sum = 0
        predictions = self.predictions_transform(predictions)
        labels = self.labels_transform(labels)
        predictions = predictions[:, : min(len(predictions[0]), self.k)]

        for prediction, label in zip(predictions, labels):
            precision = self.hits_counter(prediction, label) / len(prediction)
            precision_sum += precision

        return precision_sum / len(labels), len(labels)


class MeanAveragePrecisionAtK(AveragedMetric):
    """
    Mean average precision at k metric (MAP@k).
    """

    def __init__(self, hits_counter=ExactLabelHitCounter(), k=1, predictions_transform=lambda x: x, labels_transform=lambda x: x):
        """
        :param hits_counter: callable that counts the number of hits between a tensor of predictions and a label.
        :param k: k top results to consider.
        :param predictions_transform: transform to do on the predictions when metric is called.
        :param labels_transform: transform to do on the labels when metric is called.
        """
        super().__init__()
        self.hits_counter = hits_counter
        self.k = k
        self.predictions_transform = predictions_transform
        self.labels_transform = labels_transform

    def _calc_metric(self, predictions, labels):
        """
        :param predictions: tensor of shape (batch_size, num_predictions)
        :param labels: tensor of labels of size (batch_size,)
        :return: (MAP@k value, num samples in input)
        """
        average_precision_sum = 0
        predictions = self.predictions_transform(predictions)
        labels = self.labels_transform(labels)
        predictions = predictions[:, : min(len(predictions[0]), self.k)]

        for prediction, label in zip(predictions, labels):
            num_hits = 0
            precision_sum = 0

            for i in range(len(prediction)):
                if self.hits_counter(prediction[i: i + 1], label) > 0:
                    precision_at_cur = self.hits_counter(prediction[:i + 1], label) / (i + 1)
                    precision_sum += precision_at_cur
                    num_hits += 1

            if num_hits != 0:
                average_precision_sum += precision_sum / num_hits

        return average_precision_sum / len(labels), len(labels)
