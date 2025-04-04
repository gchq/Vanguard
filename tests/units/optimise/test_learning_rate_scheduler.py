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
Tests for `ApplyLearningRateScheduler`.
"""

import unittest

import torch

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.optimise import ApplyLearningRateScheduler
from vanguard.vanilla import GaussianGPController


class BasicTests(unittest.TestCase):
    """
    Basic tests for the `ApplyLearningRateScheduler` decorator.
    """

    def setUp(self):
        """Initialise data shared between tests."""
        self.rng = get_default_rng()
        self.dataset = SyntheticDataset(rng=self.rng)

    def test_learning_rate_is_stepped(self) -> None:
        """Test that the learning rate is modified by the scheduler."""
        num_iters = 33
        step_size = 10
        gamma = 0.9
        initial_lr = 0.2

        expected_lr = initial_lr * gamma ** (num_iters // step_size)

        @ApplyLearningRateScheduler(torch.optim.lr_scheduler.StepLR, step_size=step_size, gamma=gamma)
        class StepLRAdam(torch.optim.Adam):
            pass

        controller = GaussianGPController(
            self.dataset.train_x,
            self.dataset.train_y,
            ScaledRBFKernel,
            self.dataset.train_y_std,
            optimiser_class=StepLRAdam,
            optim_kwargs={"lr": initial_lr},
            rng=self.rng,
        )
        controller.fit(num_iters)

        # TODO: this does look like a _lot_ of protected accesses - review this test?
        # https://github.com/gchq/Vanguard/issues/207
        # pylint: disable=protected-access
        current_lr = controller._smart_optimiser._internal_optimiser._applied_scheduler.get_lr()[0]
        self.assertAlmostEqual(current_lr, expected_lr)

    def test_scheduler_handles_loss_schedules(self):
        """
        Test that the decorator handles schedulers that take the loss as a parameter.

        Some schedulers (e.g. `StepLR` in `test_learning_rate_is_stepped`) take no parameters in their step() method.
        Some, like, `ReduceLROnPlateau`, take a required "metrics" parameter.
        This test checks that the decorator can handle both cases.
        """

        @ApplyLearningRateScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
        class StepLRAdam(torch.optim.Adam):
            pass

        controller = GaussianGPController(
            self.dataset.train_x,
            self.dataset.train_y,
            ScaledRBFKernel,
            self.dataset.train_y_std,
            optimiser_class=StepLRAdam,
            optim_kwargs={"lr": 0.2},
            rng=self.rng,
        )

        # Assertion: this doesn't fail due to some "unexpected argument" or "argument missing" error
        controller.fit(10)

    def test_scheduler_handles_only_expected_type_errors(self):
        """
        Test that if the internal optimiser raises a `TypeError` other than "missing 'loss' argument", it is not caught.
        """

        @ApplyLearningRateScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
        class StepLRAdam(torch.optim.Adam):
            def step(self, *args, **kwargs):
                """Raises a `TypeError` other than "missing 'loss' argument", which should not be handled."""
                raise TypeError("Test error")

        controller = GaussianGPController(
            self.dataset.train_x,
            self.dataset.train_y,
            ScaledRBFKernel,
            self.dataset.train_y_std,
            optimiser_class=StepLRAdam,
            optim_kwargs={"lr": 0.2},
            rng=self.rng,
        )

        # Check that the error isn't suppressed
        with self.assertRaisesRegex(TypeError, "Test error"):
            controller.fit(10)
