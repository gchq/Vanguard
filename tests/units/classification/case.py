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
Contains the ClassificationTestCase class.
"""

import unittest
from typing import Union

import numpy as np
import numpy.typing
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from torch import Tensor


class BatchScaledRBFKernel(ScaleKernel):
    """
    The recommended starting place for a kernel.
    """

    def __init__(self, batch_shape: Union[int, torch.Size]) -> None:
        batch_shape = batch_shape if isinstance(batch_shape, torch.Size) else torch.Size([batch_shape])
        super().__init__(RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape)


class BatchScaledMean(ZeroMean):
    """
    A basic mean with batch shape to match the above kernel.
    """

    def __init__(self, batch_shape: Union[int, torch.Size]) -> None:
        batch_shape = batch_shape if isinstance(batch_shape, torch.Size) else torch.Size([batch_shape])
        super().__init__(batch_shape=batch_shape)


class ClassificationTestCase(unittest.TestCase):
    """
    A base class for classification tests.
    """

    @staticmethod
    def assertPredictionsEqual(  # pylint: disable=invalid-name
        x: Union[Tensor, numpy.typing.NDArray[np.floating]],
        y: Union[Tensor, numpy.typing.NDArray[np.floating]],
        delta: Union[float, int] = 0,
    ) -> None:
        """
        Assert true if predictions are correct.

        :param x: The first set of predictions.
        :param y: The second set of predictions.
        :param delta: The proportion of elements which can be outside of the interval.
        """
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)

        if x.shape != y.shape:
            raise RuntimeError(f"Shape {x.shape} does not match {y.shape}")
        number_incorrect = torch.sum(x != y)
        proportion_incorrect = number_incorrect / len(x)
        if proportion_incorrect > delta:
            error_message = (
                f"Incorrect predictions: {number_incorrect} / {len(x)} "
                f"({100 * proportion_incorrect:.2f}%) -- delta = {100 * delta:.2f}%"
            )
            if __debug__:
                raise AssertionError(error_message) from None
            else:
                raise AssertionError("Proportion of incorrect predictions bigger than the threshold value.") from None
