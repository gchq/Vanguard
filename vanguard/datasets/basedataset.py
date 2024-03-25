"""
Base datasets for Vanguard.

For the ease of the user, Vanguard contains a number of datasets commonly referenced in examples, and used in tests.
The dataset instances allow for easy access to the training and testing data through attributes.
"""
import os
from contextlib import contextmanager
from typing import Generator, Union

import numpy as np
import urllib3
from numpy.typing import NDArray
from urllib3 import BaseHTTPResponse


class Dataset:
    """
    Represents an experimental dataset used by Vanguard.
    """

    train_x: NDArray[np.floating]
    train_x_std: Union[float, NDArray[np.floating]]
    train_y: NDArray[np.floating]
    train_y_std: Union[float, NDArray[np.floating]]
    test_x: NDArray[np.floating]
    test_x_std: Union[float, NDArray[np.floating]]
    test_y: NDArray[np.floating]
    test_y_std: Union[float, NDArray[np.floating]]
    significance: float

    def __init__(
        self,
        train_x: NDArray[np.floating],
        train_x_std: Union[float, NDArray[np.floating]],
        train_y: NDArray[np.floating],
        train_y_std: Union[float, NDArray[np.floating]],
        test_x: NDArray[np.floating],
        test_x_std: Union[float, NDArray[np.floating]],
        test_y: NDArray[np.floating],
        test_y_std: Union[float, NDArray[np.floating]],
        significance: float,
    ):
        """
        Initialise self.

        :param numpy.ndarray[float] train_x: The training inputs.
        :param numpy.ndarray[float],float train_x_std: The standard deviation(s) of the training inputs.
        :param numpy.ndarray[float] train_y: The training outputs.
        :param numpy.ndarray[float],float train_y_std: The standard deviation(s) of the training outputs.
        :param numpy.ndarray[float] test_x: The test inputs.
        :param numpy.ndarray[float],float test_x_std: The standard deviation(s) of the test inputs.
        :param numpy.ndarray[float] test_y: The test outputs.
        :param numpy.ndarray[float],float test_y_std: The standard deviation(s) of the test outputs.
        :param float significance: The recommended significance value to be used for confidence intervals.
            Note that this value does not necessarily have any bearing on the data.
        """
        self.train_x = train_x
        self.train_x_std = train_x_std
        self.train_y = train_y
        self.train_y_std = train_y_std
        self.test_x = test_x
        self.test_x_std = test_x_std
        self.test_y = test_y
        self.test_y_std = test_y_std
        self.significance = significance

    @property
    def num_features(self) -> int:
        """Return the number of features."""
        return self.train_x.shape[1]

    @property
    def num_training_points(self) -> int:
        """Return the number of training points."""
        return self.train_x.shape[0]

    @property
    def num_testing_points(self) -> int:
        """Return the number of testing points."""
        return self.test_x.shape[0]

    @property
    def num_points(self) -> int:
        """Return the number of data points."""
        return self.num_training_points + self.num_testing_points


class FileDataset(Dataset):
    """
    A Vanguard dataset which requires a file to be loaded.

    If missing, this file can be
    downloaded with the :py:meth:`~vanguard.datasets.basedataset.FileDataset.download` method.
    """
    @classmethod
    def download(cls):
        """Download the data needed for this dataset."""
        raise NotImplementedError

    @staticmethod
    @contextmanager
    def _large_file_downloader(url) -> Generator[BaseHTTPResponse, None, None]:
        """Download a file within a context manager."""
        http = urllib3.PoolManager()
        request = http.request("GET", url, preload_content=False)
        try:
            yield request
        finally:
            request.release_conn()

    @staticmethod
    def _get_data_path(file_name) -> str:
        """
        Get the full path to the file name within the data folder.

        .. note::
            This will also create the ``data`` folder if it is missing. If the
            download fails then this can result in an unexpected empty folder.
        """
        current_directory_path = os.path.dirname(__file__)
        data_path = os.path.join(current_directory_path, "data")
        os.makedirs(data_path, exist_ok=True)
        full_file_path = os.path.join(data_path, file_name)
        return full_file_path


class EmptyDataset(Dataset):
    """
    Represents an empty dataset.
    """
    def __init__(self):
        """
        Initialise self.
        """
        super().__init__(np.array([]), np.array([]), np.array([]), np.array([]),
                         np.array([]), np.array([]), np.array([]), np.array([]),
                         significance=0)

    @classmethod
    def download(cls):
        """Download the data needed for this dataset."""
        raise TypeError("Not implemented for this class.")
