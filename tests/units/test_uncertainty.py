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
Tests for handling uncertainty.
"""

import unittest

import torch

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.optimise import NoImprovementError
from vanguard.uncertainty import GaussianUncertaintyGPController


class TestGaussianUncertaintyGPController(unittest.TestCase):
    """Tests for the `GaussianUncertaintyGPController`."""

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.rng = get_default_rng()
        self.dataset = SyntheticDataset(rng=self.rng)
        self.controller = GaussianUncertaintyGPController(
            train_x=self.dataset.train_x,
            train_x_std=self.dataset.train_x_std,
            train_y=self.dataset.train_y,
            y_std=self.dataset.train_y_std,
            kernel_class=ScaledRBFKernel,
            rng=self.rng,
        )

    def test_infinite_generator(self) -> None:
        """Test the training data generator created within `GaussianUncertaintyGPController`."""
        # Verify the first output from the infinite generator is as expected
        generated_data = next(self.controller.train_data_generator)
        self.assertListEqual(list(generated_data[0].shape), list(self.dataset.train_x.shape))
        self.assertListEqual(list(generated_data[1].shape), list(self.dataset.train_x_std.shape))
        self.assertListEqual(list(generated_data[2].shape), [self.dataset.train_x.shape[0]])
        self.assertListEqual(list(generated_data[3].shape), list(self.dataset.train_x.shape))
        self.assertEqual(len(generated_data), 4)

    def test_gradient_variance(self) -> None:
        """Test the gradient variance exists."""
        # Before fitting the gradient variance should exist but be None
        self.assertIsNone(self.controller.gradient_variance)

        # Fit the controller, which should now compute a gradient variance
        self.controller.fit(10)

        # After fitting, the gradient variance should have one entry per data-point, but
        # have expanded dimensions due to torch usage.
        self.assertListEqual(list(self.controller.gradient_variance.shape), [1, self.dataset.train_x.shape[0], 1])

    def test_predict_at_point(self) -> None:
        """
        Test the `predict_at_point` method.

        We should not be able to generate predictions at points as we have to account for the
        uncertainty, so should need to use `predict_at_fuzzy_point`.
        """
        with self.assertRaises(TypeError):
            self.controller.predict_at_point(self.dataset.test_x)

    def test_sgd_round_cases(self) -> None:
        """Test output of `_sgd_round` when we have no improvement in loss."""

        # Force the single optimisation step to return an error
        def mocked_single_optimisation_step(
            x: torch.Tensor,
            y: torch.Tensor,
            retain_graph: bool = False,
        ):
            raise NoImprovementError

        # pylint: disable=protected-access
        self.controller._smart_optimiser.last_n_losses = [1, 2, 3]

        # Having forced a no improvement error, we should now just get the loss as the last value
        # in self.controller._smart_optimiser.last_n_losses, that is 3
        self.controller._single_optimisation_step = mocked_single_optimisation_step
        result = self.controller._sgd_round(n_iters=10)
        # pylint: enable=protected-access
        self.assertEqual(result, 3)

    def test_sgd_round_run_time_error(self) -> None:
        """Test output of _sgd_round when we have a runtime error."""

        def mocked_single_optimisation_step(
            x: torch.Tensor,
            y: torch.Tensor,
            retain_graph: bool = False,
        ):
            raise RuntimeError

        # Having forced a run time error but not auto-restarting, the method should just fail
        # pylint: disable=protected-access
        self.controller._single_optimisation_step = mocked_single_optimisation_step
        with self.assertRaises(RuntimeError):
            self.controller._sgd_round(n_iters=1)

        # If we force an auto restart, we should eventually hit the case where it is trying to
        # restart from num_iters - 1 = 0 and crash rather than silently fail. The fail
        # manifests as a ValueError from trying to slice with an invalid input.
        self.controller.auto_restart = True
        with self.assertRaises(ValueError):
            self.controller._sgd_round(n_iters=5)
        # pylint: enable=protected-access

    def test_set_requires_grad(self) -> None:
        """Test the method _set_requires_grad."""
        # In setup we did not specify tuning of noise, so without adjustment we should have no
        # requirement for gradient on the noise
        self.assertFalse(self.controller.train_x_std.requires_grad)

        # If we call _set_requires_grad with True, we still should not require gradients because
        # we haven't told the controller to tune the noise
        # pylint: disable=protected-access
        self.controller._set_requires_grad(value=True)
        # pylint: enable=protected-access
        self.assertFalse(self.controller.train_x_std.requires_grad)

        # If we now tell the controller to tune the noise and call _set_requires_grad, we should
        # see a requirement for gradients
        # pylint: disable=protected-access
        self.controller._learn_input_noise = True
        self.controller._set_requires_grad(value=True)
        # pylint: enable=protected-access
        self.assertTrue(self.controller.train_x_std.requires_grad)

    def test_process_x_std(self) -> None:
        """Test the processing of standard deviation on inputs."""
        # pylint: disable=protected-access
        result = self.controller._process_x_std(std=None)
        # pylint: enable=protected-access

        # We expect a tensor with one element per input value given to the controller at creation time,
        # and for there to be a requirement on gradients (since we want to update this). Note the starting
        # noise value is set to 0.01.
        torch.testing.assert_close(result, torch.tensor([0.01]))
        self.assertTrue(result.requires_grad)
