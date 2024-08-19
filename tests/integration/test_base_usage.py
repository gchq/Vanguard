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
Basic end to end functionality test for Vanguard.
"""

import unittest

import numpy as np

from tests.cases import get_default_rng
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController


class VanguardTestCase(unittest.TestCase):
    """
    A subclass of TestCase designed to check end-to-end usage of base code.
    """

    def setUp(self) -> None:
        """
        Define data shared across tests.
        """
        self.rng = get_default_rng()
        self.num_train_points = 500
        self.num_test_points = 500
        self.n_sgd_iters = 100
        self.small_noise = 0.1
        self.confidence_interval_alpha = 0.9
        # When generating confidence intervals, how far from the expected number of
        # points must we empirically observe to be we willing to consider a test a
        # failure? As an example, if we have 90% confidence interval, we might expect
        # 10% of points to lie outside of this, 5% above and 5% below if everything is
        # symmetric. However, we expect some noise due to errors and finite datasets, so
        # we would only consider the test a failure if more than
        # 5% + accepted_confidence_interval_error lie above the upper confidence
        # interval
        self.accepted_confidence_interval_error = 3
        self.expected_percent_outside_one_sided = (100.0 * (1 - self.confidence_interval_alpha)) / 2

    def test_basic_gp(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable regression problem.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        GP can be fit to this data. We check that the confidence intervals are ordered
        correctly, and they contain the expected number of points in both the training
        and testing data.
        """
        # Define some data for the test
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.squeeze(x * np.sin(x))

        # Split data into training and testing
        train_indices = self.rng.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # Define the controller object, with an assumed small amount of noise
        gp = GaussianGPController(
            train_x=x[train_indices],
            train_y=y[train_indices],
            kernel_class=ScaledRBFKernel,
            y_std=self.small_noise * np.ones_like(y[train_indices]),
            rng=self.rng,
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        posterior = gp.predictive_likelihood(x)
        prediction_means, _prediction_covariances = posterior.prediction()
        _prediction_ci_median, prediction_ci_lower, prediction_ci_upper = posterior.confidence_interval(
            alpha=self.confidence_interval_alpha
        )

        # Sense check the outputs
        self.assertTrue(np.all(prediction_means <= prediction_ci_upper))
        self.assertTrue(np.all(prediction_means >= prediction_ci_lower))

        # Are the prediction intervals reasonable?
        pct_above_ci_upper_train = (
            100.0 * np.sum(y[train_indices] >= prediction_ci_upper[train_indices]) / x[train_indices].shape[0]
        )
        pct_above_ci_upper_test = (
            100.0 * np.sum(y[test_indices] >= prediction_ci_upper[test_indices]) / x[test_indices].shape[0]
        )
        pct_below_ci_lower_train = (
            100.0 * np.sum(y[train_indices] <= prediction_ci_lower[train_indices]) / x[train_indices].shape[0]
        )
        pct_below_ci_lower_test = (
            100.0 * np.sum(y[test_indices] <= prediction_ci_lower[test_indices]) / x[test_indices].shape[0]
        )
        for pct_check in [
            pct_above_ci_upper_train,
            pct_above_ci_upper_test,
            pct_below_ci_lower_train,
            pct_below_ci_lower_test,
        ]:
            self.assertLessEqual(
                pct_check, self.expected_percent_outside_one_sided + self.accepted_confidence_interval_error
            )


if __name__ == "__main__":
    unittest.main()
