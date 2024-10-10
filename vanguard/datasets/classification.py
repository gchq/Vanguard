# © Crown Copyright GCHQ
#
# Licensed under the GNU General Public License, version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The following datasets allow for straightforward experiments with synthetic classification data.
"""

import itertools
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from matplotlib.colors import Colormap
from numpy.typing import NDArray
from sklearn.datasets import make_gaussian_quantiles
from torch import Tensor

from vanguard import utils
from vanguard.datasets.basedataset import Dataset


class BinaryStripeClassificationDataset(Dataset):
    """
    Dataset comprised of one-dimensional input values, and binary output values (0 or 1).

    .. plot::

        import matplotlib.pyplot as plt
        from vanguard.datasets.classification import BinaryStripeClassificationDataset
        DATASET = BinaryStripeClassificationDataset(30, 50)
        plt.plot(DATASET.train_x, DATASET.train_y, label="Truth")
        plt.show()
    """

    def __init__(self, num_train_points: int, num_test_points: int, rng: Optional[np.random.Generator] = None) -> None:
        """
        Initialise self.

        :param num_train_points: The number of training points.
        :param num_test_points: The number of testing points.
        :param rng: Generator instance used to generate random numbers.
        """
        self.rng = utils.optional_random_generator(rng)
        train_x = np.linspace(0, 1, num_train_points).reshape((-1, 1))
        test_x = self.rng.random(num_test_points).reshape((-1, 1))

        train_y = self.even_split(train_x)
        test_y = self.even_split(test_x)

        super().__init__(train_x, np.array([]), train_y, np.array([]), test_x, np.array([]), test_y, np.array([]), 0.0)

    @staticmethod
    def even_split(x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the reals, divided into two distinct values."""
        return (np.sign(np.cos(x * (4 * np.pi))) + 1) / 2


class MulticlassGaussianClassificationDataset(Dataset):
    """
    A multiclass dataset based on :func:`sklearn.datasets.make_gaussian_quantiles`.

    .. plot::

        import matplotlib.pyplot as plt
        from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
        DATASET = MulticlassGaussianClassificationDataset(1000, 1000, num_classes=5)
        DATASET.plot()
        plt.show()
    """

    def __init__(
        self,
        num_train_points: int,
        num_test_points: int,
        num_classes: int,
        *,
        covariance_scale: float = 1.0,
        num_features: int = 2,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Initialise self.

        :param num_train_points: The number of training points.
        :param num_test_points: The number of testing points.
        :param num_classes: The number of classes.
        :param num_features: The number of features to generate for the input data.
        :param covariance_scale: The covariance matrix will be this value times the unit matrix.
            Defaults to 1.0.
        :param rng: Generator instance used to generate random numbers.
        """
        self.rng = utils.optional_random_generator(rng)
        self.num_classes = num_classes

        train_x, train_y = make_gaussian_quantiles(
            cov=covariance_scale,
            n_samples=num_train_points,
            n_features=num_features,
            n_classes=num_classes,
            random_state=self.rng.integers(2**32 - 1),
        )
        test_x, test_y = make_gaussian_quantiles(
            cov=covariance_scale,
            n_samples=num_test_points,
            n_features=num_features,
            n_classes=num_classes,
            random_state=self.rng.integers(2**32 - 1),
        )

        super().__init__(train_x, 0.0, train_y, 0.0, test_x, 0.0, test_y, 0.0, 0.0)

    @property
    def one_hot_train_y(self) -> Tensor:
        """
        Return the training data as a one-hot encoded array.

        Note that if there are exactly two classes, this returns `train_y.reshape((-1, 1))` instead.
        """
        numpy_train_y = self.train_y.detach().cpu().numpy()
        one_hot = sklearn.preprocessing.LabelBinarizer().fit_transform(numpy_train_y)
        return torch.as_tensor(one_hot, device=self.train_y.device)

    def plot(self, cmap: Union[str, Colormap] = "Set1", alpha: float = 0.5) -> None:  # pragma: no cover
        """
        Plot the data.

        :param cmap: The colour map to be used.
        :param alpha: The transparency of the points.
        """
        ax = plt.gca()
        scatter = plt.scatter(
            self.train_x[:, 0].numpy(force=True),
            self.train_x[:, 1].numpy(force=True),
            c=self.train_y.numpy(force=True),
            cmap=cmap,
            alpha=alpha,
        )
        plt.scatter(
            self.test_x[:, 0].numpy(force=True),
            self.test_x[:, 1].numpy(force=True),
            c=self.test_y.numpy(force=True),
            cmap=cmap,
            alpha=alpha,
        )
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend)

    def plot_prediction(
        self, prediction: Union[NDArray, Tensor], cmap: Union[str, Colormap] = "Set1", alpha: float = 0.5
    ) -> None:  # pragma: no cover
        """
        Plot a prediction.

        :param prediction: The predicted classes.
        :param cmap: The colour map to be used.
        :param alpha: The transparency of the points.
        """
        prediction = torch.as_tensor(prediction)
        correct_prediction = prediction == self.test_y
        proportion_correct: float = correct_prediction.sum() / len(self.test_x)  # type: ignore

        ax = plt.gca()
        correct_scatter = plt.scatter(
            self.test_x[correct_prediction, 0].numpy(force=True),
            self.test_x[correct_prediction, 1].numpy(force=True),
            c=prediction[correct_prediction].numpy(force=True),
            cmap=cmap,
            alpha=alpha,
        )
        incorrect_scatter = plt.scatter(
            self.test_x[~correct_prediction, 0].numpy(force=True),
            self.test_x[~correct_prediction, 1].numpy(force=True),
            c=prediction[~correct_prediction].numpy(force=True),
            cmap=cmap,
            marker="x",
            alpha=alpha,
        )
        legend_correct = ax.legend(*correct_scatter.legend_elements(), title="Correct", loc="upper left")
        legend_incorrect = ax.legend(*incorrect_scatter.legend_elements(), title="Incorrect", loc="lower right")
        ax.add_artist(legend_correct)
        ax.add_artist(legend_incorrect)
        plt.title(f"Proportion correct: {100 * proportion_correct:.2f}%")

    def plot_confusion_matrix(
        self, prediction: NDArray, cmap: Union[str, Colormap] = "OrRd", text_size: str = "xx-large"
    ) -> None:  # pragma: no cover
        """
        Plot a confusion matrix based on a specific prediction.

        :param prediction: The predicted classes.
        :param cmap: The colour map to be used.
        :param text_size: The text size to be used for the labels.
        """
        matrix = np.zeros((self.num_classes, self.num_classes))
        for true_label, predicted_label in zip(self.test_y, prediction):
            matrix[true_label, predicted_label] += 1

        matrix /= matrix.sum(axis=1)

        ax = plt.gca()
        ax.matshow(matrix, cmap=cmap)
        for x, y in itertools.product(range(self.num_classes), repeat=2):
            ax.text(x=x, y=y, s=str(matrix[y, x]), va="center", ha="center", size=text_size)
        plt.xlabel("Predicted classes")
        plt.ylabel("True classes")


class BinaryGaussianClassificationDataset(MulticlassGaussianClassificationDataset):
    """
    A binary dataset based on :func:`sklearn.datasets.make_gaussian_quantiles`.

    .. plot::

        import matplotlib.pyplot as plt
        from vanguard.datasets.classification import BinaryGaussianClassificationDataset
        DATASET = BinaryGaussianClassificationDataset(50, 50)
        DATASET.plot()
        plt.show()
    """

    def __init__(
        self,
        num_train_points: int,
        num_test_points: int,
        *,
        covariance_scale: float = 1.0,
        num_features: int = 2,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Initialise self.

        :param num_train_points: The number of training points.
        :param num_test_points: The number of testing points.
        :param covariance_scale: The covariance matrix will be this value times the unit matrix.
            Defaults to 1.0.
        :param num_features: The number of features to generate for the input data.
        :param rng: Generator instance used to generate random numbers.
        """
        super().__init__(
            num_train_points,
            num_test_points,
            num_classes=2,
            covariance_scale=covariance_scale,
            num_features=num_features,
            rng=utils.optional_random_generator(rng),
        )
