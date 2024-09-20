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
Basic end to end functionality test for warping of Gaussian processes in Vanguard.
"""

import unittest

import numpy as np
import torch

from tests.cases import get_default_rng_override_seed
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.warps import SetWarp
from vanguard.warps.warpfunctions import (
    AffineWarpFunction,
    ArcSinhWarpFunction,
    BoxCoxWarpFunction,
    LogitWarpFunction,
    PositiveAffineWarpFunction,
    SinhWarpFunction,
    SoftPlusWarpFunction,
)


class VanguardTestCase(unittest.TestCase):
    """
    A subclass of TestCase designed to check end-to-end usage of warping code.
    """

    def setUp(self) -> None:
        """
        Define data shared across tests.
        """
        # fails on previous seed values of 1_234, 1_989 - TODO: This is a BUG, see linked issue
        # https://github.com/gchq/Vanguard/issues/273
        self.rng = get_default_rng_override_seed(1_000_000_000)
        self.num_train_points = 50
        self.num_test_points = 50
        self.n_sgd_iters = 10
        self.small_noise = 0.1

    def test_affine_positive_affine_box_cox_arcsinh_warp(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable regression problem.

        Warping is applied, where we consider the warping functions: AffineWarpFunction,
        PositiveAffineWarpFunction, BoxCoxWarpFunction, ArcSinhWarpFunction and SinhWarpFunction.
        These are tested together as they can be applied to the same candidate data without modification,
        so reduces code duplication.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        warped GP can be fit to this data.
        """
        # Define some data
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.squeeze(x * np.sin(x)) + self.rng.normal(scale=self.small_noise, size=x.shape[0])

        # Split data into training and testing
        train_indices = self.rng.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # Consider multiple different warping functions
        for warp_function in [
            AffineWarpFunction(),
            PositiveAffineWarpFunction(b=6.0),
            BoxCoxWarpFunction(lambda_=0.5),
            ArcSinhWarpFunction(),
            SinhWarpFunction(),
        ]:
            # Define the warped controller object
            @SetWarp(warp_function, ignore_all=True)
            class WarpedController(GaussianGPController):
                pass

            # Define the controller object, with an assumed small amount of noise
            gp = WarpedController(
                train_x=x[train_indices],
                train_y=y[train_indices],
                kernel_class=ScaledRBFKernel,
                y_std=self.small_noise * np.ones_like(y[train_indices]),
                rng=self.rng,
            )

            # Fit the GP
            gp.fit(n_sgd_iters=self.n_sgd_iters)

            # Get predictions from the controller object
            prediction_medians, prediction_ci_lower, prediction_ci_upper = gp.posterior_over_point(
                x[test_indices]
            ).confidence_interval()

            # Sense check the outputs
            self.assertTrue(torch.all(prediction_medians <= prediction_ci_upper))
            self.assertTrue(torch.all(prediction_medians >= prediction_ci_lower))

    def test_soft_plus_warp(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable regression problem.

        We apply soft-plus warping using the warp function SoftPlusWarpFunction.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        GP can be fit to this data.
        """
        # Define some data - note that for numerical reasons, we must avoid certain values of `y`.
        # This is due to SoftPlusWarpFunction, which applies the warp :math:`y\mapsto\log(e^y - 1)`,
        # meaning we don't want `y` to grow too large or we might hit numerical issues when taking the exponential,
        # but also we need to ensure that :math:`e^y - 1` does not get too close to zero or become negative. For this
        # reason, we ensure `y` takes values around 2-3 which covers both cases.
        x = np.linspace(start=4, stop=6, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.squeeze(x / 2.0) + self.rng.normal(scale=self.small_noise, size=x.shape[0])

        # Split data into training and testing
        train_indices = self.rng.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # Define the warped controller object
        @SetWarp(SoftPlusWarpFunction(), ignore_all=True)
        class WarpedController(GaussianGPController):
            pass

        # Define the controller object, with an assumed small amount of noise
        gp = WarpedController(
            train_x=x[train_indices],
            train_y=y[train_indices],
            kernel_class=ScaledRBFKernel,
            y_std=self.small_noise * np.ones_like(y[train_indices]),
            rng=self.rng,
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        prediction_medians, prediction_ci_lower, prediction_ci_upper = gp.posterior_over_point(
            x[test_indices]
        ).confidence_interval()

        # Sense check the outputs
        self.assertTrue(torch.all(prediction_medians <= prediction_ci_upper))
        self.assertTrue(torch.all(prediction_medians >= prediction_ci_lower))

        # Also try to specify the gp with invalid `y` data that should not allow such warping,
        # and check an appropriate error is raised
        gp_invalid = WarpedController(
            train_x=x[train_indices],
            train_y=-100.0 * y[train_indices],
            kernel_class=ScaledRBFKernel,
            y_std=self.small_noise * np.ones_like(y[train_indices]),
            rng=self.rng,
        )
        # TODO: check for something more specific than just `Exception`!
        # https://github.com/gchq/Vanguard/issues/401
        with self.assertRaises(Exception):
            gp_invalid.fit(n_sgd_iters=self.n_sgd_iters)

    def test_logit_warp(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable regression problem.

        We apply logit warping using the warp function LogitWarpFunction.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        GP can be fit to this data.
        """
        # Define some data - note that for numerical reasons, we keep `y` between 0 and
        # 1, which ensures the logits make sense
        x = np.linspace(start=0.1, stop=1.0, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.squeeze(x / 2.0) + self.rng.normal(scale=0.1 * self.small_noise, size=x.shape[0])
        y = np.clip(y, 0, 1)

        # Split data into training and testing
        train_indices = self.rng.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # Define the warped controller object
        @SetWarp(LogitWarpFunction(), ignore_all=True)
        class WarpedController(GaussianGPController):
            pass

        # Define the controller object, with an assumed small amount of noise
        gp = WarpedController(
            train_x=x[train_indices],
            train_y=y[train_indices],
            kernel_class=ScaledRBFKernel,
            y_std=0.1 * self.small_noise * np.ones_like(y[train_indices]),
            rng=self.rng,
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        prediction_medians, prediction_ci_lower, prediction_ci_upper = gp.posterior_over_point(
            x[test_indices]
        ).confidence_interval()

        # Sense check the outputs
        assert not any([np.isnan(el) for el in prediction_ci_lower])
        assert not any([np.isnan(el) for el in prediction_medians])
        assert not any([np.isnan(el) for el in prediction_ci_upper])
        np.testing.assert_array_less(prediction_medians, prediction_ci_upper)
        np.testing.assert_array_less(prediction_ci_lower, prediction_medians)

        # Also try to specify the gp with invalid `y` data that should not allow such warping,
        # and check an appropriate error is raised
        gp_invalid = WarpedController(
            train_x=x[train_indices],
            train_y=-100.0 * y[train_indices],
            kernel_class=ScaledRBFKernel,
            y_std=self.small_noise * np.ones_like(y[train_indices]),
            rng=self.rng,
        )
        # TODO: check for something more specific than just `Exception`!
        # https://github.com/gchq/Vanguard/issues/401
        with self.assertRaises(Exception):
            gp_invalid.fit(n_sgd_iters=self.n_sgd_iters)


if __name__ == "__main__":
    unittest.main()
