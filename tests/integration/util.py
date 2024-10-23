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

"""Utility functions for use in integration tests."""

from typing import Literal, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

TrainTestData = Union[tuple[NDArray, NDArray, NDArray, NDArray], tuple[Tensor, Tensor, Tensor, Tensor]]


def train_test_split_convert(
    x: NDArray, y: NDArray, *, n_test_points: int, array_type: Literal["ndarray", "tensor"], rng: np.random.Generator
) -> TrainTestData:
    """
    Split data into train and test sets, and then convert it to either Tensors or NDArrays.

    :param x: The inputs to split.
    :param y: The targets to split.
    :param n_test_points: The number of test points.
    :param array_type: Whether to output NDArrays or Tensors.
    :param rng: Generator instance for consistent random numbers.
    """
    test_indices = rng.choice(np.arange(y.shape[0]), size=n_test_points, replace=False)
    train_indices = np.setdiff1d(np.arange(y.shape[0]), test_indices)

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return (
        convert_array_type(x_train, array_type),
        convert_array_type(x_test, array_type),
        convert_array_type(y_train, array_type),
        convert_array_type(y_test, array_type),
    )


def convert_array_type(arr: NDArray, array_type: Literal["ndarray", "tensor"]) -> Union[Tensor, NDArray]:
    """
    Convert an NDArray to a Tensor, or check that it is in fact an NDArray.

    :param arr: The array to convert.
    :param array_type: The type (tensor/ndarray) to convert to.
    """
    if array_type == "ndarray":
        assert isinstance(arr, np.ndarray)
        return arr
    else:
        assert array_type == "tensor"
        return torch.as_tensor(arr)
