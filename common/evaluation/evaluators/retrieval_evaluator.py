import torch

from .evaluator import MetricsEvaluator, Evaluator
from .. import metrics as metrics
from ...utils import module as module_utils
from ...utils import tensor as tensor_utils


class RetrievalValidationEvaluator(Evaluator):
    """
    Validation evaluator for retrieval task. Supports metrics that receive retrieved gallery indices and query indices.
    """

    def __init__(self, model, val_query_dataset, val_gallery_dataset, val_metric_info_seq=None,
                 train_query_dataset=None, train_gallery_dataset=None, train_metric_info_seq=None,
                 num_results=30, ranking_metric=tensor_utils.cosine_similarity, largest=True,
                 device=torch.device("cpu"), batch_size=256):
        self.val_metric_infos = metrics.metric_info_seq_to_dict(val_metric_info_seq) if val_metric_info_seq is not None else {}
        self.train_metric_infos = metrics.metric_info_seq_to_dict(train_metric_info_seq) if train_metric_info_seq is not None else {}

        self.metric_infos = {}
        self.metric_infos.update(self.val_metric_infos)
        self.metric_infos.update(self.train_metric_infos)
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metrics)

        self.val_metrics = {name: metric_info.metric for name, metric_info in self.val_metric_infos.items()}
        self.train_metrics = {name: metric_info.metric for name, metric_info in self.train_metric_infos.items()}
        self.metrics = {}
        self.metrics.update(self.val_metrics)
        self.metrics.update(self.train_metrics)

        self.model = model
        self.val_query_dataset = val_query_dataset
        self.val_gallery_dataset = val_gallery_dataset
        self.train_query_dataset = train_query_dataset
        self.train_gallery_dataset = train_gallery_dataset

        self.num_results = num_results
        self.ranking_metric = ranking_metric
        self.largest = largest
        self.device = device
        self.batch_size = batch_size

    def get_metric_infos(self):
        return self.metric_infos

    def get_metrics(self):
        return self.metrics

    def get_tracked_values(self):
        return self.tracked_values

    def evaluate(self):
        with torch.no_grad():
            metric_values = {}

            self.__update_metric_values(metric_values, self.val_query_dataset, self.val_gallery_dataset, self.val_metrics)
            if self.train_query_dataset is not None and self.train_gallery_dataset is not None:
                self.__update_metric_values(metric_values, self.train_query_dataset, self.train_gallery_dataset, self.train_metrics)

            return metric_values

    def __update_metric_values(self, metric_values, query_dataset, gallery_dataset, metrics):
        if len(query_dataset) == 0 or len(gallery_dataset) == 0:
            return

        calculated_metric_values = self.__calc_retrieval_metrics(self.model, query_dataset, gallery_dataset, metrics,
                                                                 num_results=self.num_results,
                                                                 ranking_metric=self.ranking_metric, largest=self.largest,
                                                                 device=self.device,
                                                                 feature_calc_batch_size=self.batch_size)
        metric_values.update(calculated_metric_values)

    @staticmethod
    def __calc_retrieval_metrics(model, query_dataset, gallery_dataset, metrics, num_results=30,
                                 ranking_metric=tensor_utils.cosine_similarity, largest=True, feature_calc_batch_size=256,
                                 device=torch.device("cpu")):
        """
        Calculates and returns a dictionary with the values of the given validation metrics.
        :param model: model to extract features from the images.
        :param query_dataset: image dataset of the query images in the validation set.
        :param gallery_dataset: image dataset of the gallery images to retrieve similar images from in the validation set.
        :param metrics: dictionary of metrics where the key is the metric name and the value is a callable to calculate the metric. The metric should
        receive 2 parameters (gallery_results_indices, query_indices) and return the metric value.
        :param num_results: number of results to retrieve.
        :param ranking_metric: similarity metric to retrieve similar images.
        :param largest: true if the most similar are those with the largest metric value, false for smallest.
        :param feature_calc_batch_size: size of batch to calculate feature vectors in.
        :param device: device to calculate features on.
        """
        query_features = module_utils.predict_in_batches(model, query_dataset, batch_size=feature_calc_batch_size, device=device)
        gallery_features = module_utils.predict_in_batches(model, gallery_dataset, batch_size=feature_calc_batch_size, device=device)
        return RetrievalValidationEvaluator.__calc_retrieval_metrics_for_query_gallery_feature_vectors(query_features, gallery_features, metrics,
                                                                                                       num_results=num_results,
                                                                                                       ranking_metric=ranking_metric, largest=largest)

    @staticmethod
    def __calc_retrieval_metrics_for_query_gallery_feature_vectors(query_features, gallery_features, metrics, num_results=30,
                                                                   ranking_metric=tensor_utils.cosine_similarity, largest=True):
        """
        Calculates and returns a dictionary with the values of the given validation metrics for the given precalculated features. For each query
        vectors the results will be taken from the gallery vectors.
        :param query_features: extracted features of the query images.
        :param gallery_features: extracted features of the gallery images.
        :param metrics: dictionary of metrics where the key is the metric name and the value is a callable to calculate the metric. The metric should
        receive 2 parameters (gallery_results_indices, query_indices) and return the metric value.
        :param num_results: number of results to retrieve.
        :param ranking_metric: similarity metric to retrieve similar images.
        :param largest: true if the most similar are those with the largest metric value, false for smallest.
        """
        _, query_results = tensor_utils.get_top_results(query_features, gallery_features, num_results=num_results, metric=ranking_metric,
                                                        largest=largest)
        metric_values = {}
        for metric_name, metric in metrics.items():
            metric_values[metric_name] = metric(query_results, range(len(query_results)))

        return metric_values
