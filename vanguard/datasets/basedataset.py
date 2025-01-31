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

from typing import Union

import numpy as np
import torch
from numpy.typing import NDArray


class Dataset:
    """
    Represents an experimental dataset used by Vanguard.

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

    def __init__(
        self,
        train_x: Union[NDArray[np.floating], torch.Tensor],
        train_x_std: Union[float, NDArray[np.floating], torch.Tensor],
        train_y: Union[NDArray[np.floating], NDArray[np.integer], torch.Tensor],
        train_y_std: Union[float, NDArray[np.floating], torch.Tensor],
        test_x: Union[NDArray[np.floating], torch.Tensor],
        test_x_std: Union[float, NDArray[np.floating], torch.Tensor],
        test_y: Union[NDArray[np.floating], NDArray[np.integer], torch.Tensor],
        test_y_std: Union[float, NDArray[np.floating], torch.Tensor],
        significance: float,
    ) -> None:
        """Initialise self."""
        self.train_x = torch.as_tensor(train_x)
        self.train_x_std = torch.as_tensor(train_x_std)
        self.train_y = torch.as_tensor(train_y)
        self.train_y_std = torch.as_tensor(train_y_std)
        self.test_x = torch.as_tensor(test_x)
        self.test_x_std = torch.as_tensor(test_x_std)
        self.test_y = torch.as_tensor(test_y)
        self.test_y_std = torch.as_tensor(test_y_std)
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


class EmptyDataset(Dataset):
    """
    Represents an empty dataset.

    :param num_features: The number of features to give the dataset. (The dataset does not contain any points,
        but the arrays `train_x` and `test_x` will have shape `(0, num_features)` to enable code that expects a
        sensible `num_features` to work.)
    :param significance: The recommended significance value to be used for confidence intervals.
        Note that this value has no bearing on the data, as there is no data - this parameter is only provided
        for compatibility with code that requires a certain significance level.
    """

    def __init__(self, num_features: int = 1, significance: float = 0.1) -> None:
        """Initialise an empty dataset."""
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
