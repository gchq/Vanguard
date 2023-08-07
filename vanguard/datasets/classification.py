"""
The following datasets allow for straightforward experiments with synthetic classification data.
"""
from functools import reduce
import itertools
from operator import or_
import pickle

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import confusion_matrix

from .basedataset import Dataset, FileDataset


class BinaryStripeClassificationDataset(Dataset):
    """
    Dataset comprised of one-dimensional input values, and binary output values.

    .. plot::

        import matplotlib.pyplot as plt
        from vanguard.datasets.classification import BinaryStripeClassificationDataset
        DATASET = BinaryStripeClassificationDataset(30, 50)
        plt.plot(DATASET.train_x, DATASET.train_y, label="Truth")
        plt.show()
    """
    def __init__(self, num_train_points, num_test_points):
        """
        Initialise self.

        :param int num_train_points: The number of training points.
        :param num_test_points: The number of testing points.
        """
        train_x = np.linspace(0, 1, num_train_points)
        test_x = np.random.rand(num_test_points)

        train_y = self.even_split(train_x)
        test_y = self.even_split(test_x)

        super().__init__(train_x, np.array([]), train_y, np.array([]),
                         test_x, np.array([]), test_y, np.array([]), 0)

    @staticmethod
    def even_split(x):
        """Return the reals, divided into two distinct values."""
        return (np.sign(np.cos(x * (4 * np.pi))) + 1) / 2


class MulticlassGaussianClassificationDataset(Dataset):
    """
    A multiclass dataset based on :py:func:`sklearn.datasets.make_gaussian_quantiles`.

    .. plot::

        import matplotlib.pyplot as plt
        from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
        DATASET = MulticlassGaussianClassificationDataset(1000, 1000, num_classes=5)
        DATASET.plot()
        plt.show()
    """
    def __init__(self, num_train_points, num_test_points, num_classes, covariance_scale=1.0, seed=None):
        """
        Initialise self.

        :param int num_train_points: The number of training points.
        :param int num_test_points: The number of testing points.
        :param int num_classes: The number of classes.
        :param float,int covariance_scale: The covariance matrix will be this value times the unit matrix.
            Defaults to 1.0.
        :param int,None seed: Used to seed the quantile creation, defaults to None (not reproducible).
        """
        self.num_classes = num_classes

        train_x, train_y = make_gaussian_quantiles(cov=covariance_scale, n_samples=num_train_points,
                                                   n_features=2, n_classes=num_classes, random_state=seed)
        test_x, test_y = make_gaussian_quantiles(cov=covariance_scale, n_samples=num_test_points,
                                                 n_features=2, n_classes=num_classes, random_state=seed)

        super().__init__(train_x, 0, train_y, 0,
                         test_x, 0, test_y, 0, 0)

    @property
    def one_hot_train_y(self):
        """Return the training data as a one-hot encoded array."""
        return sklearn.preprocessing.LabelBinarizer().fit_transform(self.train_y)

    def plot(self, cmap="Set1", alpha=0.5):
        """
        Plot the data.

        :param str cmap: The colour map to be used.
        :param float alpha: The transparency of the points.
        """
        ax = plt.gca()
        scatter = plt.scatter(self.train_x[:, 0], self.train_x[:, 1], c=self.train_y, cmap=cmap, alpha=alpha)
        plt.scatter(self.test_x[:, 0], self.test_x[:, 1], c=self.test_y, cmap=cmap, alpha=alpha)
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend)

    def plot_prediction(self, prediction, cmap="Set1", alpha=0.5):
        """
        Plot a prediction.

        :param numpy.ndarray prediction: The predicted classes.
        :param str cmap: The colour map to be used.
        :param float alpha: The transparency of the points.
        :param int point_size: The size of each individual point.
        :param int edge_width: The width of the edge of each point.
        """
        correct_prediction = (prediction == self.test_y)
        proportion_correct = correct_prediction.sum() / len(self.test_x)

        ax = plt.gca()
        correct_scatter = plt.scatter(self.test_x[correct_prediction, 0], self.test_x[correct_prediction, 1],
                                      c=prediction[correct_prediction], cmap=cmap, alpha=alpha)
        incorrect_scatter = plt.scatter(self.test_x[~correct_prediction, 0], self.test_x[~correct_prediction, 1],
                                        c=prediction[~correct_prediction], cmap=cmap, marker="x", alpha=alpha)
        legend_correct = ax.legend(*correct_scatter.legend_elements(), title="Correct", loc="upper left")
        legend_incorrect = ax.legend(*incorrect_scatter.legend_elements(), title="Incorrect", loc="lower right")
        ax.add_artist(legend_correct)
        ax.add_artist(legend_incorrect)
        plt.title(f"Proportion correct: {100 * proportion_correct:.2f}%")

    def plot_confusion_matrix(self, prediction, cmap="OrRd", text_size="xx-large"):
        """
        Plot a confusion matrix based on a specific prediction.

        :param numpy.ndarray prediction: The predicted classes.
        :param str cmap: The colour map to be used.
        :param str text_size: The text size to be used for the labels.
        """
        matrix = np.zeros((self.num_classes, self.num_classes))
        for true_label, predicted_label in zip(self.test_y, prediction):
            matrix[true_label, predicted_label] += 1

        matrix /= matrix.sum(axis=1)

        ax = plt.gca()
        ax.matshow(matrix, cmap=cmap)
        for x, y in itertools.product(range(self.num_classes), repeat=2):
            ax.text(x=x, y=y, s=matrix[y, x], va="center", ha="center", size=text_size)
        plt.xlabel("Predicted classes")
        plt.ylabel("True classes")


class BinaryGaussianClassificationDataset(MulticlassGaussianClassificationDataset):
    """
    A binary dataset based on :py:func:`sklearn.datasets.make_gaussian_quantiles`.

    .. plot::

        import matplotlib.pyplot as plt
        from vanguard.datasets.classification import BinaryGaussianClassificationDataset
        DATASET = BinaryGaussianClassificationDataset(50, 50)
        DATASET.plot()
        plt.show()
    """
    def __init__(self, num_train_points, num_test_points, covariance_scale=1.0, seed=None):
        """
        Initialise self.

        :param int num_train_points: The number of training points.
        :param num_test_points: The number of testing points.
        :param float,int covariance_scale: The covariance matrix will be this value times the unit matrix.
            Defaults to 1.0.
        :param int,None seed: Used to seed the quantile creation, defaults to None (not reproducible).
        """
        super().__init__(num_train_points, num_test_points, num_classes=2, covariance_scale=covariance_scale, seed=seed)
