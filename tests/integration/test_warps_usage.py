"""
Basic end to end functionality test for warping of Gaussian processes in Vanguard.
"""

import unittest

import numpy as np

from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.warps import SetWarp
from vanguard.warps.warpfunctions import (
    AffineWarpFunction,
    ArcSinhWarpFunction,
    BoxCoxWarpFunction,
    LogitWarpFunction,
    PositiveAffineWarpFunction,
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
        self.random_seed = 1_989
        self.num_train_points = 500
        self.num_test_points = 500
        self.n_sgd_iters = 100
        self.small_noise = 0.1

    def test_affine_positive_affine_box_cox_arcsinh_warp(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable regression problem.

        Warping is applied, where we consider the warping functions: AffineWarpFunction,
        PositiveAffineWarpFunction, BoxCoxWarpFunction and ArcSinhWarpFunction. These are
        tested together as they can be applied to the same candidate data without modification,
        so reduces code duplication.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        warped GP can be fit to this data.
        """
        np.random.seed(self.random_seed)

        # Define some data
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.squeeze(x * np.sin(x)) + np.random.normal(scale=self.small_noise, size=x.shape[0])

        # Split data into training and testing
        train_indices = np.random.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # Consider multiple different warping functions
        for warp_function in [
            AffineWarpFunction(),
            PositiveAffineWarpFunction(b=6.0),
            BoxCoxWarpFunction(lambda_=0.5),
            ArcSinhWarpFunction(),
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
            )

            # Fit the GP
            gp.fit(n_sgd_iters=self.n_sgd_iters)

            # Get predictions from the controller object
            prediction_medians, prediction_ci_lower, prediction_ci_upper = gp.posterior_over_point(
                x[test_indices]
            ).confidence_interval()

            # Sense check the outputs
            self.assertTrue(np.all(prediction_medians <= prediction_ci_upper))
            self.assertTrue(np.all(prediction_medians >= prediction_ci_lower))

    def test_soft_plus_warp(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable regression problem.

        We apply soft-plus warping using the warp function SoftPlusWarpFunction.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        GP can be fit to this data.
        """
        np.random.seed(self.random_seed)

        # Define some data - note that for numerical reasons, we keep `y` close 2 and
        # 3, which ensures we don't take huge exponents or logs of negative numbers
        x = np.linspace(start=4, stop=6, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.squeeze(x / 2.0) + np.random.normal(scale=self.small_noise, size=x.shape[0])

        # Split data into training and testing
        train_indices = np.random.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
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
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        prediction_medians, prediction_ci_lower, prediction_ci_upper = gp.posterior_over_point(
            x[test_indices]
        ).confidence_interval()

        # Sense check the outputs
        self.assertTrue(np.all(prediction_medians <= prediction_ci_upper))
        self.assertTrue(np.all(prediction_medians >= prediction_ci_lower))

    def test_logit_warp(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable regression problem.

        We apply logit warping using the warp function LogitWarpFunction.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        GP can be fit to this data.
        """
        np.random.seed(self.random_seed)

        # Define some data - note that for numerical reasons, we keep `y` between 0 and
        # 1, which ensures the logits make sense
        x = np.linspace(start=0.1, stop=1.0, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.squeeze(x / 2.0) + np.random.normal(scale=0.1 * self.small_noise, size=x.shape[0])

        # Split data into training and testing
        train_indices = np.random.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
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
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        prediction_medians, prediction_ci_lower, prediction_ci_upper = gp.posterior_over_point(
            x[test_indices]
        ).confidence_interval()

        # Sense check the outputs
        self.assertTrue(np.all(prediction_medians <= prediction_ci_upper))
        self.assertTrue(np.all(prediction_medians >= prediction_ci_lower))


if __name__ == "__main__":
    unittest.main()
