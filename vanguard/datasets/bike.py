"""
The bike dataset contains messy information about bike rentals, and is a good dataset for testing performance.
"""


import os
import shutil
import tempfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .basedataset import FileDataset


class BikeDataset(FileDataset):
    """
    Comparison of bike rentals to weather information.

    Contains the hourly count of rental bikes between years 2011 and 2012 in Capital bikeshare system with the
    corresponding weather and seasonal information. Supplied by the Machine Learning Repository [Bike]_.
    """
    # TODO: Change this link
    __DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"

    def __init__(self, n_samples=None, training_proportion=0.9, significance=0.025, noise_scale=0.001, seed=42):
        """
        Initialise self.

        :param int,None n_samples: The number of samples to use. If None, all samples will be used.
        :param float training_proportion: The proportion of data used for training, defaults to 0.9.
        :param float significance: The significance used, defaults to 0.025.
        :param float noise_scale: The standard deviation of a given vector v is taken to be
            ``noise_scale * np.abs(v).mean()``. Defaults to 0.001.
        :param int,None seed: The seed for the model, defaults to 42.
        """
        data = self._load_data()
        np.random.seed(seed)
        np.random.shuffle(data)

        n_samples = self._get_n_samples(data, n_samples)
        x = data[:n_samples, :-1]
        y = data[:n_samples, -1]

        x_std = noise_scale * np.abs(x).mean()
        y_std = noise_scale * np.abs(y).mean()

        y /= y.mean()

        n_train = int(training_proportion * x.shape[0])
        train_x, test_x = x[:n_train], x[n_train:]
        train_y, test_y = y[:n_train], y[n_train:]

        train_x_std, test_x_std = np.ones_like(train_x)*x_std, np.ones_like(test_x) * x_std
        train_y_std, test_y_std = np.ones_like(train_y)*y_std, np.ones_like(test_y) * y_std

        super().__init__(train_x, train_x_std, train_y, train_y_std,
                         test_x, test_x_std, test_y, test_y_std,
                         significance)

    def plot(self):
        """
        Plot the data.

        :param float deviate: Defines the size of the error bars to be plotted.
        :param float alpha: The transparency of the error bars.
        """
        raise NotImplementedError("Dataset plotting not implemented for bike data.")

    def plot_prediction(self, pred_y_mean, pred_y_lower, pred_y_upper, y_upper_bound=None, error_width=0.3):
        """
        Plot a prediction using its confidence interval.

        :param float deviate: Defines the size of the error bars to be plotted.
        :param float alpha: The transparency of the error bars.
        """
        keep_indices = (self.test_y < y_upper_bound) if y_upper_bound else np.ones_like(self.test_y, dtype=bool)

        plot_y = self.test_y[keep_indices]
        plot_upper = pred_y_upper[keep_indices]
        plot_lower = pred_y_lower[keep_indices]
        plot_mean = pred_y_mean[keep_indices]

        rmse = np.sqrt(np.mean((plot_y - plot_mean)**2))

        plt.errorbar(plot_y, plot_mean, yerr=np.vstack([plot_mean - plot_lower, plot_upper - plot_mean]),
                     marker="o", label="mean", linestyle="", markersize=1, elinewidth=error_width)
        plt.plot(plot_y, plot_y, color="black", linestyle="dotted", linewidth=1, alpha=0.7)
        plt.xlabel("True y values")
        plt.ylabel("Predicted y values")
        plt.legend()
        plt.title(f"RMSE: {rmse:.4f}")

    def plot_y(self, start=0, stop=5, num_samples=1_000):
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

    def _load_data(self):
        """Load the data."""
        file_path = self._get_data_path("bike.csv")
        try:
            df = pd.read_csv(file_path, parse_dates=['dteday'])
        except FileNotFoundError:
            message = (f"Could not find data at {file_path}. If you have not downloaded the data, "
                       f"call {type(self).__name__}.download().")
            raise FileNotFoundError(message)
        # Extract the day of the date and convert it to an integer
        df['dteday'] = df['dteday'].apply(lambda x: int(x.strftime('%d')))
        # Instant is just an index and casual+registered = count
        df.drop(columns=['instant', 'casual', 'registered'], inplace=True)
        data = df.values
        return data

    @staticmethod
    def _get_n_samples(data, n_samples):
        """Get samples from the data."""
        if n_samples is None:
            n_samples = data.shape[0]
        if n_samples > data.shape[0]:
            print(f'You requested {n_samples} samples but the data is of length {data.shape[0]}. '
                  f'Returning {data.shape[0]} samples instead.')
        return n_samples

    @classmethod
    def download(cls):
        """Download the dataset."""
        final_bike_path = cls._get_data_path("bike.csv")

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_file_path = os.path.join(temp_dir, "bike.zip")
            with cls._large_file_downloader(cls.__DOWNLOAD_URL) as request:
                with open(zip_file_path, "wb") as wf:
                    for chunk in request.stream(32):
                        wf.write(chunk)

            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            hours_file = os.path.join(temp_dir, "hour.csv")
            shutil.copy(hours_file, final_bike_path)
