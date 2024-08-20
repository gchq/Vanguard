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
The bike dataset contains messy information about bike rentals, and is a good dataset for testing performance.

Supplied by the UC Irvine Machine Learning Repository :cite:`FanaeeT2013`.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from vanguard.datasets.basedataset import FileDataset
import vanguard.utils as utils


class BikeDataset(FileDataset):
    """
    Comparison of bike rentals to weather information.

    Contains the hourly count of rental bikes between years 2011 and 2012 in Capital bikeshare system with the
    corresponding weather and seasonal information. Supplied by the UC Irvine Machine Learning Repository
    :cite:`FanaeeT2013`.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        training_proportion: float = 0.9,
        significance: float = 0.025,
        noise_scale: float = 0.001,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialise self.

        :param num_samples: The number of samples to use. If None, all samples will be used.
        :param training_proportion: The proportion of data used for training, defaults to 0.9.
        :param significance: The significance used, defaults to 0.025.
        :param noise_scale: The standard deviation of a given vector v is taken to be
            ``noise_scale * np.abs(v).mean()``. Defaults to 0.001.
        :param rng: Generator instance used to generate random numbers.
        """
        data = self._load_data()

        self.rng = utils.optional_random_generator(rng)
        self.rng.shuffle(data)

        num_samples = self._get_n_samples(data, num_samples)
        x = data[:num_samples, :-1]
        y = data[:num_samples, -1]

        x_std = noise_scale * np.abs(x).mean()
        y_std = noise_scale * np.abs(y).mean()

        y /= y.mean()

        n_train = int(training_proportion * x.shape[0])
        train_x, test_x = x[:n_train], x[n_train:]
        train_y, test_y = y[:n_train], y[n_train:]

        train_x_std, test_x_std = np.ones_like(train_x) * x_std, np.ones_like(test_x) * x_std
        train_y_std, test_y_std = np.ones_like(train_y) * y_std, np.ones_like(test_y) * y_std

        super().__init__(
            train_x, train_x_std, train_y, train_y_std, test_x, test_x_std, test_y, test_y_std, significance
        )

    def plot(self) -> None:  # pragma: no cover
        """
        Plot the data.
        """
        raise NotImplementedError("Dataset plotting not implemented for bike data.")

    def plot_prediction(
        self,
        pred_y_mean: np.typing.NDArray[np.floating],
        pred_y_lower: np.typing.NDArray[np.floating],
        pred_y_upper: np.typing.NDArray[np.floating],
        y_upper_bound: Optional[np.typing.NDArray[np.floating]] = None,
        error_width: float = 0.3,
    ) -> None:  # pragma: no cover
        """
        Plot a prediction using its confidence interval.

        :param pred_y_mean: Array of prediction means.
        :param pred_y_lower: Lower bound of predictions, e.g. from a prediction interval.
        :param pred_y_upper: Upper bound of predictions, e.g. from a prediction interval.
        :param y_upper_bound: If provided, any points in the test set above this value will be discarded from plotting.
        :param error_width: Error bar line width.
        """
        keep_indices = (self.test_y < y_upper_bound) if y_upper_bound else np.ones_like(self.test_y, dtype=bool)

        plot_y = self.test_y[keep_indices]
        plot_upper = pred_y_upper[keep_indices]
        plot_lower = pred_y_lower[keep_indices]
        plot_mean = pred_y_mean[keep_indices]

        rmse = np.sqrt(np.mean((plot_y - plot_mean) ** 2))

        plt.errorbar(
            plot_y,
            plot_mean,
            yerr=np.vstack([plot_mean - plot_lower, plot_upper - plot_mean]),
            marker="o",
            label="mean",
            linestyle="",
            markersize=1,
            elinewidth=error_width,
        )
        plt.plot(plot_y, plot_y, color="black", linestyle="dotted", linewidth=1, alpha=0.7)
        plt.xlabel("True y values")
        plt.ylabel("Predicted y values")
        plt.legend()
        plt.title(f"RMSE: {rmse:.4f}")

    def plot_y(self, start: int = 0, stop: int = 5, num_samples: int = 1_000) -> None:  # pragma: no cover
        """
        Visualize the target variable.

        :param float start: The start of the y-values to be plotted, defaults to 0.
        :param float stop: The end of the y-values to be plotted, defaults to 5.
        :param int num_samples: The number of samples to be plotted, defaults to 1,000.
        """
        x = np.linspace(start, stop, num_samples)

        plt.plot(x, stats.gaussian_kde(self.train_y)(x), label="train")
        plt.plot(x, stats.gaussian_kde(self.test_y)(x), label="test")

        plt.grid(alpha=0.5)
        plt.ylabel("density", fontsize=15)
        plt.xlabel("$y$", fontsize=15)
        plt.legend()

    def _load_data(self) -> pd.DataFrame:
        """Load the data."""
        file_path = self._get_data_path("bike.csv")
        try:
            df = pd.read_csv(file_path, parse_dates=["dteday"])
        except FileNotFoundError as exc:
            message = (f"Could not find data at {file_path}.")
            raise FileNotFoundError(message) from exc
        # Extract the day of the date and convert it to an integer
        df["dteday"] = df["dteday"].apply(lambda x: int(x.strftime("%d")))
        # Instant is just an index and casual+registered = count
        df.drop(columns=["instant", "casual", "registered"], inplace=True)
        data = df.values
        return data

    @staticmethod
    def _get_n_samples(data: pd.DataFrame, n_samples: Optional[int]) -> int:
        """
        Verify the number of samples is valid for some given data.

        :param data: Dataframe we wish to take samples from.
        :param n_samples: The number of samples we wish to take.
        :return: A number representing how many samples to take, potentially altered based on its
            relation to the size of the provided data.
        """
        if n_samples is None:
            n_samples = data.shape[0]
        if n_samples > data.shape[0]:
            warnings.warn(
                "You requested more samples than there are datapoints in the data. Using "
                "all datapoints instead."
            )
            n_samples = data.shape[0]
        if n_samples < 0:
            raise ValueError('A negative number of samples has been requested.')
        return n_samples
