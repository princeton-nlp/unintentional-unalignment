import os

from .callback import *
from ...utils import visualization as visualization_utils


class MetricsPlotter(Callback):
    """
    Creates figures for visualization of training progress and metrics and saves them to files.
    """

    DEFAULT_FOLDER_NAME = "plots"

    def __init__(self, output_dir: str, folder_name: str = DEFAULT_FOLDER_NAME, create_dir: bool = True, create_plots_interval: int = 1,
                 exclude: Sequence[str] = None):
        """
        :param output_dir: directory for saved plots folder.
        :param folder_name: folder name under output_dir to save plots to.
        :param create_dir: create output directory if is not exist.
        :param create_plots_interval: interval of epochs to plot metrics.
        :param exclude: sequence of metric names to exclude.
        """
        self.output_dir = output_dir
        self.folder_name = folder_name
        self.create_dir = create_dir

        self.create_plots_interval = create_plots_interval
        self.exclude = exclude if exclude is not None else set()

        self.plots_dir = os.path.join(self.output_dir, self.folder_name)

    def on_fit_initialization(self, trainer):
        if self.create_dir and not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def on_epoch_end(self, trainer):
        if (trainer.epoch + 1) % self.create_plots_interval == 0:
            self.__create_plots(trainer.train_evaluator.get_metric_infos_with_history(), trainer.train_evaluator.get_tracked_values_with_history(),
                                trainer.val_evaluator.get_metric_infos_with_history(), trainer.val_evaluator.get_tracked_values_with_history(),
                                trainer.value_store.get_tracked_values_with_history())

    def on_fit_end(self, trainer, num_epochs_ran, fit_output):
        self.__create_plots(trainer.train_evaluator.get_metric_infos_with_history(), trainer.train_evaluator.get_tracked_values_with_history(),
                            trainer.val_evaluator.get_metric_infos_with_history(), trainer.val_evaluator.get_tracked_values_with_history(),
                            trainer.value_store.get_tracked_values_with_history())

    @staticmethod
    def __escape_metric_name(metric_name):
        return metric_name.lower().replace(" ", "_")

    def __create_plots(self, train_metric_infos, train_tracked_values, val_metric_infos, val_tracked_values, other_tracked_values):
        aggregated_by_tag_tracked_values = self.__get_aggregated_tracked_values_by_tag(train_metric_infos, train_tracked_values,
                                                                                       val_metric_infos, val_tracked_values, other_tracked_values)

        for tag, tracked_values_dict in aggregated_by_tag_tracked_values.items():
            x_values = []
            y_values = []
            line_labels = []
            for metric_plot_name, tracked_value in tracked_values_dict.items():
                if len(tracked_value.epochs_with_values) == 0:
                    continue

                x_values.append(tracked_value.epochs_with_values)
                y_values.append(tracked_value.epoch_values)
                line_labels.append(metric_plot_name)

            if len(x_values) == 0:
                continue

            fig = visualization_utils.create_line_plot_figure(x_values, y_values, title=tag,
                                                              xlabel="epoch", ylabel=tag,
                                                              line_labels=line_labels)
            escaped_tag = self.__escape_metric_name(tag)
            fig.savefig(os.path.join(self.plots_dir, f"{escaped_tag}.png"))

    def __get_aggregated_tracked_values_by_tag(self, train_metric_infos, train_tracked_values, val_metric_infos, val_tracked_values,
                                               other_tracked_values):
        """
        Returns dict of tag to dict of metric name to tracked value. The metric names have the phase added as a prefix to avoid ambiguity.
        """
        aggregated_by_tag_tracked_values = {}
        self.__populate_by_tag_tracked_values(aggregated_by_tag_tracked_values, train_metric_infos, train_tracked_values)
        self.__populate_by_tag_tracked_values(aggregated_by_tag_tracked_values, val_metric_infos, val_tracked_values)
        self.__populate_by_tag_other_tracked_values(aggregated_by_tag_tracked_values, other_tracked_values)
        return aggregated_by_tag_tracked_values

    def __populate_by_tag_tracked_values(self, aggregated_by_tag_tracked_values, metric_infos, tracked_values):
        metric_infos = {name: metric_info for name, metric_info in metric_infos.items() if name not in self.exclude and metric_info.is_scalar}
        tracked_values = {name: tracked_value for name, tracked_value in tracked_values.items() if name in metric_infos}

        for metric_name, metric_info in metric_infos.items():
            if metric_info.tag not in aggregated_by_tag_tracked_values:
                aggregated_by_tag_tracked_values[metric_info.tag] = {}

            aggregated_by_tag_tracked_values[metric_info.tag][metric_name] = tracked_values[metric_name]

    def __populate_by_tag_other_tracked_values(self, aggregated_by_tag_tracked_values, other_tracked_values):
        tracked_values = {name: tracked_value for name, tracked_value in other_tracked_values.items() if
                          name not in self.exclude and tracked_value.is_scalar}

        for name, tracked_value in tracked_values.items():
            tag = name
            if tag not in aggregated_by_tag_tracked_values:
                aggregated_by_tag_tracked_values[tag] = {}

            aggregated_by_tag_tracked_values[tag][name] = tracked_value
