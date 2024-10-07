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

from typing import Literal, Optional

import numpy as np
import pytest
import torch
from gpytorch.mlls import VariationalELBO

from tests.cases import get_default_rng
from tests.integration.util import convert_array_type, train_test_split_convert
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference


@VariationalInference()
class VariationalController(GaussianGPController):
    """Variational controller for testing."""


class TestBaseUsage:
    num_train_points = 500
    num_test_points = 500
    n_sgd_iters = 100
    small_noise = 0.1
    confidence_interval_alpha = 0.9
    # When generating confidence intervals, how far from the expected number of
    # points must we empirically observe to be we willing to consider a test a
    # failure? As an example, if we have 90% confidence interval, we might expect
    # 10% of points to lie outside of this, 5% above and 5% below if everything is
    # symmetric. However, we expect some noise due to errors and finite datasets, so
    # we would only consider the test a failure if more than
    # 5% + accepted_confidence_interval_error lie above the upper confidence
    # interval
    accepted_confidence_interval_error = 3
    expected_percent_outside_one_sided = (100.0 * (1 - confidence_interval_alpha)) / 2

    @pytest.mark.parametrize(
        "batch_size",
        [
            pytest.param(None, id="full"),
            # Currently, confidence intervals on a `@VariationalInference`-decorated controller are really inaccurate.
            # This was an issue before batched training on exact GPs was forbidden, but it's come up now because batch
            # training requires the use of `@VariationalInference`.
            pytest.param(
                100,
                id="batched",
                marks=[pytest.mark.xfail(reason="Variational confidence intervals are currently inaccurate.")],
            ),
        ],
    )
    @pytest.mark.parametrize("array_type", ["tensor", "ndarray"])
    def test_basic_gp(self, batch_size: Optional[int], array_type: Literal["tensor", "ndarray"]) -> None:
        """
        Verify Vanguard usage on a simple, single variable regression problem.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        GP can be fit to this data. We check that the confidence intervals are ordered
        correctly, and they contain the expected number of points in both the training
        and testing data.

        We test this both in and out of batch mode.
        """
        # Define some data for the test
        rng = get_default_rng()

        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.squeeze(x * np.sin(x))

        # Split data into training and testing
        x_train, x_test, y_train, y_test = train_test_split_convert(
            x, y, n_test_points=self.num_test_points, array_type=array_type, rng=rng
        )
        y_train_std = convert_array_type(self.small_noise * np.ones_like(y_train), array_type)

        # Define the controller object, with an assumed small amount of noise
        if batch_size is None:
            gp = GaussianGPController(
                train_x=x_train,
                train_y=y_train,
                kernel_class=ScaledRBFKernel,
                y_std=y_train_std,
                rng=rng,
                batch_size=batch_size,
            )
        else:
            gp = VariationalController(
                train_x=x_train,
                train_y=y_train,
                kernel_class=ScaledRBFKernel,
                y_std=y_train_std,
                marginal_log_likelihood_class=VariationalELBO,
                rng=rng,
                batch_size=batch_size,
            )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        posterior_train = gp.predictive_likelihood(x_train)
        prediction_means_train, _ = posterior_train.prediction()
        _, prediction_ci_lower_train, prediction_ci_upper_train = posterior_train.confidence_interval(
            alpha=self.confidence_interval_alpha
        )
        posterior_test = gp.predictive_likelihood(x_test)
        prediction_means_test, _ = posterior_test.prediction()
        _, prediction_ci_lower_test, prediction_ci_upper_test = posterior_test.confidence_interval(
            alpha=self.confidence_interval_alpha
        )

        # Sense check the outputs
        assert torch.all(prediction_means_train <= prediction_ci_upper_train)
        assert torch.all(prediction_means_train >= prediction_ci_lower_train)
        assert torch.all(prediction_means_test <= prediction_ci_upper_test)
        assert torch.all(prediction_means_test >= prediction_ci_lower_test)

        # Convert inputs to tensor to avoid device confusion issues on comparison here
        y_train = torch.as_tensor(y_train)
        y_test = torch.as_tensor(y_test)

        # Are the prediction intervals reasonable?
        pct_above_ci_upper_train = 100.0 * torch.sum(y_train >= prediction_ci_upper_train) / self.num_train_points
        pct_above_ci_upper_test = 100.0 * torch.sum(y_test >= prediction_ci_upper_test) / self.num_test_points
        pct_below_ci_lower_train = 100.0 * torch.sum(y_train <= prediction_ci_lower_train) / self.num_train_points
        pct_below_ci_lower_test = 100.0 * torch.sum(y_test <= prediction_ci_lower_test) / self.num_test_points
        for pct_check in [
            pct_above_ci_upper_train,
            pct_above_ci_upper_test,
            pct_below_ci_lower_train,
            pct_below_ci_lower_test,
        ]:
            assert pct_check <= self.expected_percent_outside_one_sided + self.accepted_confidence_interval_error
