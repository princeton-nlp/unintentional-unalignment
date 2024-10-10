import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from . import tensor as tensor_utils


def tensor_imshow(tensor_image):
    """
    Given a tensor that represents an image with size (C, H, W), plots and shows the image in a new figure.
    """
    np_image = tensor_image.numpy()
    plt.figure()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    plt.show()


def show_top_query_results_plot(feature_vectors, images, query_indices=None, num_rand_queries=4, num_results=4,
                                metric=tensor_utils.cosine_similarity, largest=True):
    """
    Plots and shows an image top query result plot for the given feature vectors.
    :param feature_vectors: feature vectors
    :param images: original images from which the input tensors were created, needs to support access by index. Each image
    is PIL image.
    :param query_indices: indices of the query images or None for random query indices
    :param num_rand_queries: number of random queries to do (in case query_indices is not given)
    :param num_results: number of wanted results
    :param metric: metric to query top results by
    :param largest: whether to take largest results (False for smallest)
    """
    query_indices = query_indices if query_indices is not None else np.random.randint(len(feature_vectors),
                                                                                      size=num_rand_queries)
    query_vectors = feature_vectors[query_indices]

    metric_vals, results_indices = tensor_utils.get_top_results(query_vectors, feature_vectors, num_results=num_results,
                                                                metric=metric, largest=largest)

    show_query_result_plot(images, query_indices, results_indices.numpy(), metric_vals.numpy(), metric_name=metric.__name__)


def show_query_result_plot(images, query_indices, results_indices, metric_vals, metric_name=""):
    """
    Plots and shows an image query result plot.
    :param images: original images from which the input tensors were created, needs to support access by index. Each image
    is PIL image.
    :param query_indices: tensor with the query indices
    :param results_indices: numpy ndarray with the result indices of size (num_queries, num_results)
    :param metric_vals: numpy ndarray with metric values of size (num_queries, num_results)
    :param metric_name: optional metric name for display purposes
    """
    query_images = [images[index] for index in query_indices]
    results_images = [__get_query_result_images(images, result_indices) for result_indices in results_indices]

    show_query_result_images_plot(query_images, results_images, metric_vals, metric_name=metric_name)


def __get_query_result_images(images, result_indices):
    """
    Gets the result images.
    """
    return [images[result_index] for result_index in result_indices]


def show_query_result_images_plot(query_images, results_images, metric_vals, metric_name=""):
    """
    Given query images as a sequence of PIL images and a list of result images lists also in PIL images, plots and
    shows a query result plot.
    """
    num_queries = len(query_images)
    fig, axarr = plt.subplots(num_queries, len(results_images[0]) + 1)
    fig.suptitle(f"Query {metric_name} Top Results")

    for i in range(num_queries):
        axarr[i, 0].axis("off")
        axarr[i, 0].imshow(np.asarray(query_images[i]))
        axarr[i, 0].set_title("Query")

        for j in range(len(results_images[0])):
            axarr[i, j + 1].axis("off")
            axarr[i, j + 1].set_title("{:.4f}".format(metric_vals[i, j]))
            axarr[i, j + 1].imshow(np.asarray(results_images[i][j]))

    plt.show()


def show_image_classification_predictions(images, predicted_labels, predictions_probabilities, correct_labels, row_length=4):
    """
    Plots in a grid the given images and their predictions with probabilities.
    :param images: sequence of PIL images.
    :param predicted_labels: predicted class label or name for each image.
    :param predictions_probabilities: prediction probability for each image.
    :param correct_labels: correct class labels or name for each image.
    :param row_length: length of each image row.
    """
    num_rows = len(images) // row_length
    if len(images) % num_rows != 0:
        num_rows += 1

    fig, axarr = plt.subplots(num_rows, row_length)
    fig.suptitle("Predictions")
    for i in range(len(images)):
        row = i // row_length
        col = i % row_length
        ax = axarr[row, col]
        ax.axis("off")
        ax.set_title(f"Predicted: {predicted_labels[i]} " + "{:.4f}".format(predictions_probabilities[i]) +
                     f"\nCorrect: {correct_labels[i]}")
        ax.imshow(np.asarray(images[i]))

    plt.show()


def show_embeddings_pca(embeddings, use_3d=False):
    """
    Plots a scatter plot of the PCA embeddings for the given embeddings.
    :param embeddings: Numpy embeddings vectors of shape (N, d).
    :param use_3d: If true will show a 3d PCA embeddings scatter plot. Otherwise, uses 2d.
    """
    pca = PCA(n_components=3 if use_3d else 2)
    pca_result = pca.fit_transform(embeddings)

    fig = plt.figure()
    if use_3d:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_title("3 Component PCA")
        ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("2 Component PCA")
        ax.scatter(pca_result[:, 0], pca_result[:, 1])

    ax.grid()
    plt.show()


def show_confusion_matrix(confusion_mat, classes, normalize=False, cmap=plt.cm.Blues):
    """
    Plots the given confusion matrix.
    :param confusion_mat: LxL matrix where L is the number of labels.
    :param classes: sequence of names of the L labels.
    :param normalize: flag whether to normalize by rows the matrix (giving per class accuracy).
    :param cmap: color map for the confusion matrix.
    """
    plt.figure()

    if normalize:
        confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1, keepdims=True)

    plt.imshow(confusion_mat, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_mat.max() / 2.
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        plt.text(j, i, format(confusion_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def show_histogram(values, bins=None, title="", xlabel="Value"):
    """
    Plots a histogram for the given values.
    :param values: array or sequence of arrays with values.
    :param bins: int or sequence or str. See matplotlib hist docs for more info.
    :param title: title of the plot.
    :param xlabel: label for x axis.
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    plt.hist(values, bins=bins)
    plt.show()


def show_bar_plot(values_dict, title="", xlabel="", ylabel="", sort=True, sort_by_values=True, descending=True, display_keys=True):
    """
    Shows a bar plot for the given values in the dictionary with their keys as x labels.
    :param values_dict: dictionary of key names and values.
    :param title: figure title.
    :param xlabel: x label for the figure.
    :param ylabel: y label for the figure.
    :param sort: flag whether to sort the bars.
    :param sort_by_values: flag whether to sort by values or by keys. If True will sort by values and otherwise by keys (assuming sort is True).
    :param descending: if sorted is True, determines order of sort.
    :param display_keys: flag whether to display the keys names on the x axis or just the rank.
    """
    keys = []
    values = []
    for key, value in values_dict.items():
        keys.append(key)
        values.append(value)

    if sort:
        if sort_by_values:
            values = np.array(values)
            sorted_indices = np.argsort(-values if descending else values)
        else:
            sorted_indices = sorted(range(len(keys)), key=lambda index: -keys[index] if descending else keys[index])

        keys = [keys[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

    plt.bar(range(len(values_dict)), values, align='center')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(values_dict)), keys if display_keys else range(len(values_dict)))

    plt.show()


def create_line_plot_figure(x_values, y_values, title="", xlabel="", ylabel="", line_labels=None, markers=None, colors=None):
    """
    Creates and returns a figure with line plots for the given x and y values.
    :param x_values: sequence of x value sequences. Each element in x_values should be a sequence of x values for the matching line.
    :param y_values: sequence of y value sequences. Each element in y_values should be a sequence of y values for the matching line.
    :param title: title of the figure.
    :param xlabel: label for x axis.
    :param ylabel: label for y axis.
    :param line_labels: optional names for the lines to be plotted. Length should match x_values and y_values length.
    :param markers: optional marker values for the lines.
    :param colors: optional color values for the lines.
    :return: line plots figure.
    """
    names = line_labels if line_labels is not None else [""] * len(x_values)
    markers = markers if markers is not None else [""] * len(x_values)
    colors = colors if colors is not None else [""] * len(x_values)

    fig = create_garbage_collectable_figure()
    ax = fig.add_subplot(111)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    for x, y, name, marker, color in zip(x_values, y_values, names, markers, colors):
        if color:
            ax.plot(x, y, label=name, marker=marker, color=color)
        else:
            ax.plot(x, y, label=name, marker=marker)

    if line_labels is not None:
        ax.legend()

    return fig


def create_garbage_collectable_figure():
    """
    Creates a matplotlib figure object that is garbage collectable and doesn't need to be closed. Can be used for creating plots for saving and not
    showing.
    :return: figure object.
    """
    fig = Figure()
    FigureCanvas(fig)
    return fig
