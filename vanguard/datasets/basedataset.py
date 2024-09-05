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
Base datasets for Vanguard.

For the ease of the user, Vanguard contains a number of datasets commonly referenced in examples, and used in tests.
The dataset instances allow for easy access to the training and testing data through attributes.
"""

import os
from typing import Union

import numpy as np
from numpy.typing import NDArray


class Dataset:
    """
    Represents an experimental dataset used by Vanguard.
    """

    def __init__(
        self,
        train_x: NDArray[np.floating],
        train_x_std: Union[float, NDArray[np.floating]],
        train_y: Union[NDArray[np.floating], NDArray[np.integer]],
        train_y_std: Union[float, NDArray[np.floating]],
        test_x: NDArray[np.floating],
        test_x_std: Union[float, NDArray[np.floating]],
        test_y: Union[NDArray[np.floating], NDArray[np.integer]],
        test_y_std: Union[float, NDArray[np.floating]],
        significance: float,
    ) -> None:
        """
        Initialise self.

        :param train_x: The training inputs.
        :param train_x_std: The standard deviation(s) of the training inputs.
        :param train_y: The training outputs.
        :param train_y_std: The standard deviation(s) of the training outputs.
        :param test_x: The test inputs.
        :param test_x_std: The standard deviation(s) of the test inputs.
        :param test_y: The test outputs.
        :param test_y_std: The standard deviation(s) of the test outputs.
        :param significance: The recommended significance value to be used for confidence intervals.
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
    """

    @staticmethod
    def _get_data_path(file_name: str) -> str:
        """
        Get the full path to the file name within the data folder.

        .. note::
            This will also create the ``data`` folder if it is missing, but the data should be
            placed there manually by the user.
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

    def __init__(self, num_features: int = 1, significance: float = 0.1) -> None:
        """
        Initialise an empty dataset.

        :param num_features: The number of features to give the dataset. (The dataset does not contain any points,
            but the arrays `train_x`, `test_y` etc. will have shape `(0, num_features)` to enable code that expects a
            sensible `num_features` to work.
        :param significance: The recommended significance value to be used for confidence intervals.
            Note that this value has no bearing on the data, as there is no data - this parameter is only provided
            for compatibility with code that requires a certain significance level.
        """
        super().__init__(
            np.zeros((0, num_features)),
            np.zeros((0,)),
            np.zeros((0,)),
            np.zeros((0,)),
            np.zeros((0, num_features)),
            np.zeros((0,)),
            np.zeros((0,)),
            np.zeros((0,)),
            significance=significance,
        )
