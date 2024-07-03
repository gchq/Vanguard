"""
Basic end to end functionality test for hierarchical code in Vanguard.
"""

import unittest

import numpy as np
from gpytorch.kernels import RBFKernel

from vanguard.hierarchical import (
    BayesianHyperparameters,
    LaplaceHierarchicalHyperparameters,
    VariationalHierarchicalHyperparameters,
)
from vanguard.vanilla import GaussianGPController


class VanguardTestCase(unittest.TestCase):
    """
    A subclass of TestCase designed to check end-to-end usage of hierarchical code.
    """

    def setUp(self) -> None:
        """
        Define data shared across tests.
        """
        self.rng = np.random.default_rng(1_989)
        self.num_train_points = 100
        self.num_test_points = 100
        self.n_sgd_iters = 100
        self.small_noise = 0.05
        self.num_mc_samples = 50

        # Define data for the tests
        self.x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        self.y = np.squeeze(self.x * np.sin(self.x))

        # Split data into training and testing
        self.train_indices = self.rng.choice(np.arange(self.y.shape[0]), size=self.num_train_points, replace=False)
        self.test_indices = np.setdiff1d(np.arange(self.y.shape[0]), self.train_indices)

    def test_gp_laplace_hierarchical(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable regression problem when using
        hierarchical hyperparameters with the LaplaceHierarchicalHyperparameters decorator.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        GP can be fit to this data.
        """

        # Create a hierarchical controller and a kernel that has Bayesian hyperparameters to estimate
        @LaplaceHierarchicalHyperparameters(num_mc_samples=self.num_mc_samples)
        class HierarchicalController(GaussianGPController):
            pass

        @BayesianHyperparameters()
        class BayesianRBFKernel(RBFKernel):
            pass

        # Define the controller object
        gp = HierarchicalController(
            train_x=self.x[self.train_indices],
            train_y=self.y[self.train_indices],
            kernel_class=BayesianRBFKernel,
            y_std=self.small_noise,
            rng=self.rng,
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        prediction_medians, prediction_ci_lower, prediction_ci_upper = gp.posterior_over_point(
            self.x[self.test_indices]
        ).confidence_interval()

        # Sense check the outputs. Note that we do not check confidence interval quality here,
        # just that they can be created, due to highly varying quality of the resulting intervals,
        self.assertTrue(np.all(prediction_medians <= prediction_ci_upper))
        self.assertTrue(np.all(prediction_medians >= prediction_ci_lower))

    def test_gp_variational_hierarchical(self):
        """
        Verify Vanguard usage on a simple, single variable regression problem when using
        hierarchical hyperparameters with the VariationalHierarchicalHyperparameters decorator.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        GP can be fit to this data.
        """

        # Create a hierarchical controller and a kernel that has Bayesian hyperparameters to estimate
        @VariationalHierarchicalHyperparameters(num_mc_samples=self.num_mc_samples)
        class HierarchicalController(GaussianGPController):
            pass

        @BayesianHyperparameters()
        class BayesianRBFKernel(RBFKernel):
            pass

        # Define the controller object
        gp = HierarchicalController(
            train_x=self.x[self.train_indices],
            train_y=self.y[self.train_indices],
            kernel_class=BayesianRBFKernel,
            y_std=self.small_noise,
            rng=self.rng,
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        prediction_medians, prediction_ci_lower, prediction_ci_upper = gp.posterior_over_point(
            self.x[self.test_indices]
        ).confidence_interval()

        # Sense check the outputs. Note that we do not check confidence interval quality here,
        # just that they can be created, due to highly varying quality of the resulting intervals,
        self.assertTrue(np.all(prediction_medians <= prediction_ci_upper))
        self.assertTrue(np.all(prediction_medians >= prediction_ci_lower))


if __name__ == "__main__":
    unittest.main()
