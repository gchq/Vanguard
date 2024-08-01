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
Tests for the Multitask decorator.
"""

import unittest

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.multitask import Multitask
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference


class ErrorTests(unittest.TestCase):
    """
    Tests that the correct error messages are thrown.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.rng = get_default_rng()
        self.dataset = SyntheticDataset(rng=self.rng)

    def test_single_task_variational(self) -> None:
        """Should throw an error."""

        @Multitask(num_tasks=1)
        @VariationalInference()
        class MultitaskController(GaussianGPController):
            pass

        with self.assertRaises(TypeError):
            MultitaskController(
                self.dataset.train_x, self.dataset.train_y, ScaledRBFKernel, self.dataset.train_y_std, rng=self.rng
            )

    def test_bad_batch_shape(self) -> None:
        """Should throw an error."""

        @Multitask(num_tasks=1)
        class MultitaskController(GaussianGPController):
            pass

        with self.assertRaises(TypeError):
            MultitaskController(
                self.dataset.train_x,
                self.dataset.train_y,
                ScaledRBFKernel,
                self.dataset.train_y_std,
                kernel_kwargs={"batch_shape": 2},
                rng=self.rng,
            )
