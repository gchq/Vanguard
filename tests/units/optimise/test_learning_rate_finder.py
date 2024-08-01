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
Tests for `LearningRateFinder`.
"""

import unittest
from unittest.mock import patch

import numpy as np
from linear_operator.utils.errors import NanError

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.optimise import LearningRateFinder
from vanguard.vanilla import GaussianGPController


class BasicTests(unittest.TestCase):
    """
    Basic tests for the `LearningRateFinder` class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up data shared between all tests."""
        rng = get_default_rng()
        cls.dataset = SyntheticDataset(rng=rng)

    def setUp(self):
        """Set up the controller for each test."""
        self.rng = get_default_rng()
        self.controller = GaussianGPController(
            self.dataset.train_x, self.dataset.train_y, ScaledRBFKernel, self.dataset.train_y_std, rng=self.rng
        )

    def test_learning_rate_finder(self) -> None:
        """Test that `LearningRateFinder` finds the minimum learning rate in the given range."""
        lr_finder = LearningRateFinder(self.controller)

        min_lr = 1e-5
        max_lr = 10
        num_divisions = 25
        learning_rate_losses = {}

        # We patch out the _run_learning_rate function as we're just testing the support apparatus here. Instead,
        # we just generate a random number, and record it for testing.
        def random_record_output(learning_rate: float, *_):
            """Generate and record a random number."""
            loss = self.rng.random()
            learning_rate_losses[learning_rate] = loss
            return loss

        with patch.object(lr_finder, "_run_learning_rate", side_effect=random_record_output) as mock_learning_rate:
            lr_finder.find(num_divisions=num_divisions, start_lr=min_lr, end_lr=max_lr)

        # check we tried the given number of learning rates
        assert mock_learning_rate.call_count == 25

        # check that the best learning rate is in between the min and max
        assert min_lr <= lr_finder.best_learning_rate <= max_lr

        # check that our best learning rate is in fact the smallest one we found
        assert lr_finder.best_learning_rate in learning_rate_losses
        assert learning_rate_losses[lr_finder.best_learning_rate] == min(learning_rate_losses.values())

    def test_run_learning_rate_infinite_on_fail(self):
        """Test that if training fails due to NaNs before the max iterations is reached, the loss is infinity."""
        lr_finder = LearningRateFinder(self.controller)

        with patch.object(self.controller, "fit", side_effect=NanError):
            # pylint: disable-next=protected-access
            loss = lr_finder._run_learning_rate(lr=1, max_iterations=10)

        assert loss == np.inf

    def test_run_learning_rate_parameter_pass_through(self):
        """Test that `_run_learning_rate` correctly passes through its parameters to the controller."""
        lr_finder = LearningRateFinder(self.controller)

        learning_rate = 1234
        max_iterations = 4321

        with patch.object(self.controller, "fit") as mock_fit:
            # pylint: disable-next=protected-access
            lr_finder._run_learning_rate(lr=learning_rate, max_iterations=max_iterations)

        mock_fit.assert_called_once_with(n_sgd_iters=max_iterations)
        assert self.controller.learning_rate == learning_rate
