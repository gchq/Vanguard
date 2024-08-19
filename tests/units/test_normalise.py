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
Tests for the NormaliseY decorator.
"""

import unittest

import numpy as np
import torch

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.normalise import NormaliseY
from vanguard.vanilla import GaussianGPController


@NormaliseY(ignore_methods=("__init__",))
class NormalisedGaussianGPController(GaussianGPController):
    """Test class."""


class BasicTests(unittest.TestCase):
    """
    Basic tests for the NormaliseY decorator.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Define variables shared across tests."""
        rng = get_default_rng()
        cls.dataset = SyntheticDataset(rng=rng)

        cls.controller = NormalisedGaussianGPController(
            cls.dataset.train_x, cls.dataset.train_y, ScaledRBFKernel, cls.dataset.train_y_std, rng=rng
        )

        cls.train_y_mean = cls.dataset.train_y.mean()
        cls.train_y_std = cls.dataset.train_y.std()
        cls.controller.fit(10)

    def test_pre_normalisation(self) -> None:
        """Test that data is not normalised before applying the normalisation."""
        self.assertNotAlmostEqual(0, self.train_y_mean)
        self.assertNotAlmostEqual(1, self.train_y_std, delta=0.05)

    def test_normalisation(self) -> None:
        """Test that data has mean zero, variance one after normalisation is applied."""
        self.assertAlmostEqual(0, self.controller.train_y.mean().detach().item())
        self.assertAlmostEqual(1, self.controller.train_y.std().detach().item(), delta=0.05)

    def test_prediction_scaling(self) -> None:
        """Test that internal and external predictions are properly scaled."""
        posterior = self.controller.posterior_over_point(self.dataset.test_x)

        # protected access is ok - we're specifically reaching in to test the internals
        # pylint: disable=protected-access
        internal_mean, internal_covar = posterior._tensor_prediction()
        # pylint: enable=protected-access
        external_mean, external_covar = posterior.prediction()

        torch.testing.assert_close(torch.tensor((external_mean - self.train_y_mean) / self.train_y_std), internal_mean)
        torch.testing.assert_close(torch.tensor(external_covar / self.train_y_std**2), internal_covar)

    def test_fuzzy_prediction_scaling(self) -> None:
        """Test that internal and external fuzzy predictions are properly scaled."""
        posterior = self.controller.posterior_over_fuzzy_point(self.dataset.test_x, self.dataset.test_x_std)

        # protected access is ok - we're specifically reaching in to test the internals
        # pylint: disable=protected-access
        internal_mean, internal_covar = posterior._tensor_prediction()
        # pylint: enable=protected-access
        external_mean, external_covar = posterior.prediction()

        torch.testing.assert_close(torch.tensor((external_mean - self.train_y_mean) / self.train_y_std), internal_mean)
        torch.testing.assert_close(torch.tensor(external_covar / self.train_y_std**2), internal_covar)

    def test_confidence_interval_scaling(self) -> None:
        """Test that internal and external confidence interval predictions are properly scaled."""
        posterior = self.controller.posterior_over_point(self.dataset.test_x)

        # protected access is ok - we're specifically reaching in to test the internals
        # pylint: disable=protected-access
        internal_median, internal_upper, internal_lower = posterior._tensor_confidence_interval(0.05)
        # pylint: enable=protected-access
        external_median, external_upper, external_lower = posterior.confidence_interval(0.05)

        torch.testing.assert_close(
            torch.tensor((external_median - self.train_y_mean) / self.train_y_std), internal_median
        )
        torch.testing.assert_close(
            torch.tensor((external_upper - self.train_y_mean) / self.train_y_std), internal_upper
        )
        torch.testing.assert_close(
            torch.tensor((external_lower - self.train_y_mean) / self.train_y_std), internal_lower
        )

    def test_fuzzy_confidence_interval_scaling(self) -> None:
        """Test that internal and external fuzzy confidence interval predictions are properly scaled."""
        posterior = self.controller.posterior_over_fuzzy_point(self.dataset.test_x, self.dataset.test_x_std)

        # protected access is ok - we're specifically reaching in to test the internals
        # pylint: disable=protected-access
        internal_median, internal_upper, internal_lower = posterior._tensor_confidence_interval(0.05)
        # pylint: enable=protected-access
        external_median, external_upper, external_lower = posterior.confidence_interval(0.05)

        torch.testing.assert_close(
            torch.tensor((external_median - self.train_y_mean) / self.train_y_std), internal_median
        )
        torch.testing.assert_close(
            torch.tensor((external_upper - self.train_y_mean) / self.train_y_std), internal_upper
        )
        torch.testing.assert_close(
            torch.tensor((external_lower - self.train_y_mean) / self.train_y_std), internal_lower
        )

    def test_log_probability_scaling(self) -> None:
        """Test that log-probability computations are as expected when using normalised data."""
        # Create the posterior over the data and compute the log probability
        posterior = self.controller.posterior_over_point(self.dataset.train_x)
        log_prob_normalised = posterior.log_probability(self.dataset.train_y)

        # The above should be the same as creating a standard Gaussian process controller
        # with a specific mean and covariance, and evaluating the scale data. Verify this by
        # creating a standard controller and manually swapping out the mean vector and
        # covariance matrix
        pre_scaled_train_y = (self.dataset.train_y - self.train_y_mean) / self.train_y_std
        rng = get_default_rng()
        standard_controller = GaussianGPController(
            train_x=self.dataset.train_x,
            train_y=pre_scaled_train_y,
            kernel_class=ScaledRBFKernel,
            y_std=self.train_y_std,
            rng=rng,
        )
        standard_posterior = standard_controller.posterior_over_point(self.dataset.train_x)

        # Manually set the mean vector and covariance matrix used by the scaled controller - if
        # the controller operates correctly, we would expect only minor numerical differences between
        # using a normalised controller and this swapping method
        standard_posterior.distribution.loc = posterior.distribution.loc
        standard_posterior.distribution.lazy_covariance_matrix = posterior.distribution.lazy_covariance_matrix
        standard_log_prob = standard_posterior.log_probability(pre_scaled_train_y)

        self.assertAlmostEqual(log_prob_normalised, standard_log_prob, delta=10.0)

    def test_sample_scaling(self) -> None:
        """Test that samples generated are as expected when using normalised data."""
        # Create the posterior over the data and compute the log probability
        posterior = self.controller.posterior_over_point(self.dataset.test_x)
        samples = posterior.sample(n_samples=10_000)

        # Samples should be on the scale of the original data (pre-scaling)
        sample_ranges = np.quantile(samples, axis=0, q=[0.0, 1.0])
        self.assertTrue(np.all(sample_ranges[0, :] <= self.dataset.test_y))
        self.assertTrue(np.all(sample_ranges[1, :] >= self.dataset.test_y))

    def test_with_no_y_std(self) -> None:
        """Test normalisation functionality when no `y_std` is provided."""
        # First, define a dataset where this is no noise on the outputs, and create a controller to
        # reflect this
        single_test_rng = get_default_rng()
        dataset = SyntheticDataset(rng=single_test_rng)
        dataset.train_y_std = None
        dataset.test_y_std = None
        controller = NormalisedGaussianGPController(
            dataset.train_x, dataset.train_y, ScaledRBFKernel, y_std=None, rng=single_test_rng
        )
        controller.fit(10)

        # Now, generate predictions over the test set - here we expect the numerics to still scale the
        # data when performing computations, so check this
        posterior_test = controller.posterior_over_point(dataset.test_x)

        # protected access is ok - we're specifically reaching in to test the internals
        # pylint: disable=protected-access
        internal_mean, internal_covar = posterior_test._tensor_prediction()
        # pylint: enable=protected-access
        external_mean, external_covar = posterior_test.prediction()

        torch.testing.assert_close(
            torch.tensor((external_mean - dataset.train_y.mean()) / dataset.train_y.std()), internal_mean
        )
        torch.testing.assert_close(torch.tensor(external_covar / dataset.train_y.std() ** 2), internal_covar)

        self.assertAlmostEqual(0, controller.train_y.mean().detach().item())
        self.assertAlmostEqual(1, controller.train_y.std().detach().item(), delta=0.05)

        # As a key point of setting y_std to zero, we should have no noise on the training inputs - so the training
        # data should be predicted exactly with no uncertainty.  We verify this by creating confidence intervals for
        # the training data predictions and verifying they are sufficiently small. Note that several nan values can
        # appear in the confidence intervals when numerically we have no spread in the data - but we expect some very
        # small amount of spread in some cases due to numerics
        posterior_train_zero_y_std = controller.posterior_over_point(dataset.train_x)
        _, external_upper_zero_y_std, external_lower_zero_y_std = posterior_train_zero_y_std.confidence_interval(0.05)
        self.assertLess(np.nanmean(np.abs(external_upper_zero_y_std - external_lower_zero_y_std)), 0.01)

        # As a final sense check, if we make predictions of the training data with the controller created where there
        # is input noise, we expect the corresponding confidence intervals to be much wider than the ones created when
        # assuming no noise on the data.
        posterior_train_non_zero_y_std = self.controller.posterior_over_point(self.dataset.train_x)
        _, external_upper_non_zero_y_std, external_lower_non_zero_y_std = (
            posterior_train_non_zero_y_std.confidence_interval(0.05)
        )
        self.assertLess(
            np.nanmean(np.abs(external_upper_zero_y_std - external_lower_zero_y_std)),
            np.nanmean(np.abs(external_upper_non_zero_y_std - external_lower_non_zero_y_std)),
        )
