"""
Tests for the VariationalHierarchicalHyperparameters and BayesianHyperparameters decorators.
"""

import abc
import unittest
from typing import Any, Generic, Type, TypeVar

import numpy as np
import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import RBFKernel, ScaleKernel

from tests.cases import get_default_rng
from vanguard.base import GPController
from vanguard.datasets.synthetic import MultidimensionalSyntheticDataset, SyntheticDataset
from vanguard.hierarchical import (
    BayesianHyperparameters,
    LaplaceHierarchicalHyperparameters,
    VariationalHierarchicalHyperparameters,
)
from vanguard.vanilla import GaussianGPController

N_MC_SAMPLES = 13


@VariationalHierarchicalHyperparameters(num_mc_samples=N_MC_SAMPLES, ignore_all=True)
class VariationalFullBayesianGPController(GaussianGPController):
    pass


@LaplaceHierarchicalHyperparameters(num_mc_samples=N_MC_SAMPLES, ignore_all=True, temperature=1.0)
class LaplaceFullBayesianGPController(GaussianGPController):
    pass


@BayesianHyperparameters()
class BayesianRBFKernel(RBFKernel):
    pass


class ScaledBayesianRBFKernel(ScaleKernel):
    def __init__(self, *args: Any, batch_shape: torch.Size = torch.Size([]), **kwargs: Any) -> None:
        super().__init__(BayesianRBFKernel(*args, batch_shape=batch_shape, **kwargs), batch_shape=batch_shape)


@BayesianHyperparameters()
class BayesianScaledRBFKernel(ScaleKernel):
    def __init__(self, *args: Any, batch_shape: torch.Size = torch.Size([]), **kwargs: Any) -> None:
        super().__init__(BayesianRBFKernel(*args, batch_shape=batch_shape, **kwargs), batch_shape=batch_shape)


class KernelConversionTests(unittest.TestCase):
    def setUp(self):
        self.rng = get_default_rng()

    def test_kernel_bayesian_hyperparameters_prepared(self) -> None:
        kernel = BayesianRBFKernel()
        self.assertEqual(len(kernel.bayesian_hyperparameters), 1)
        self.assertEqual(kernel.bayesian_hyperparameters[0].raw_name, "raw_lengthscale")
        self.assertEqual(kernel.bayesian_hyperparameters[0].raw_shape, torch.Size([1, 1]))
        self.assertIsInstance(kernel.bayesian_hyperparameters[0].constraint, Positive)
        self.assertEqual(kernel.bayesian_hyperparameters[0].prior_mean, 8)
        self.assertEqual(kernel.bayesian_hyperparameters[0].prior_variance, 6**2)

    def test_kernel_bayesian_hyperparameters_prepared_ard_shape(self) -> None:
        kernel = BayesianRBFKernel(ard_num_dims=5)
        self.assertEqual(kernel.bayesian_hyperparameters[0].raw_shape, torch.Size([1, 5]))

    def test_kernel_bayesian_hyperparameters_variational_samples(self) -> None:
        dataset = SyntheticDataset(rng=self.rng)
        gp = VariationalFullBayesianGPController(
            dataset.train_x, dataset.train_y, BayesianRBFKernel, dataset.train_y_std, rng=self.rng
        )
        self.assertEqual(gp.kernel.raw_lengthscale.shape, torch.Size([N_MC_SAMPLES, 1, 1]))

    def test_dimension_of_bayesian_variational_hyperparameters(self) -> None:
        dataset = SyntheticDataset(rng=self.rng)
        gp = VariationalFullBayesianGPController(
            dataset.train_x, dataset.train_y, ScaledBayesianRBFKernel, dataset.train_y_std, rng=self.rng
        )
        self.assertEqual(gp.hyperparameter_collection.sample_tensor.shape, torch.Size([N_MC_SAMPLES, 1]))


GPControllerT = TypeVar("GPControllerT", bound=GPController)


class AbstractTests:
    # namespace the test case ABCs below so they don't get run by unittest
    class TrainingTests(unittest.TestCase, Generic[GPControllerT], metaclass=abc.ABCMeta):
        """
        Basic tests for an hierarchical controller and BayesianHyperparameters decorators.
        """

        def setUp(self):
            self.rng = get_default_rng()

        @property
        @abc.abstractmethod
        def controller_class(self) -> Type[GPControllerT]:
            """
            The GPController subclass to be tested.

            Note that this isn't actually implemented as a property in subclasses of TrainingTest - it's only a property
            here so that I can type hint it without implementing it!
            """

        def test_non_bayesian_hyperparameters_are_point_estimates(self) -> None:
            dataset = SyntheticDataset(rng=self.rng)
            gp = self.controller_class(
                dataset.train_x, dataset.train_y, ScaledBayesianRBFKernel, dataset.train_y_std, rng=self.rng
            )
            gp.fit(10)
            self.assertNotEqual(gp.kernel.raw_outputscale.item(), 0.0)

        def test_posterior_does_not_fail(self) -> None:
            dataset = SyntheticDataset(rng=self.rng)
            gp = self.controller_class(
                dataset.train_x, dataset.train_y, ScaledBayesianRBFKernel, dataset.train_y_std, rng=self.rng
            )
            gp.fit(10)
            posterior = gp.posterior_over_point(dataset.test_x)
            mean, upper, lower = posterior.confidence_interval()
            self.assertNoNans(mean)
            self.assertNoNans(upper)
            self.assertNoNans(lower)

        def test_2d_non_bayesian_hyperparameters_are_point_estimates(self) -> None:
            dataset = MultidimensionalSyntheticDataset(rng=self.rng)
            gp = self.controller_class(
                dataset.train_x,
                dataset.train_y,
                ScaledBayesianRBFKernel,
                dataset.train_y_std,
                kernel_kwargs={"ard_num_dims": 2},
                rng=self.rng,
            )
            gp.fit(10)
            self.assertNotEqual(gp.kernel.raw_outputscale.item(), 0.0)

        def test_2d_posterior_does_not_fail(self) -> None:
            dataset = MultidimensionalSyntheticDataset(rng=self.rng)
            gp = self.controller_class(
                dataset.train_x,
                dataset.train_y,
                ScaledBayesianRBFKernel,
                dataset.train_y_std,
                kernel_kwargs={"ard_num_dims": 2},
                rng=self.rng,
            )
            gp.fit(10)
            posterior = gp.posterior_over_point(dataset.test_x)
            mean, upper, lower = posterior.confidence_interval()
            self.assertNoNans(mean)
            self.assertNoNans(upper)
            self.assertNoNans(lower)

        def assertNoNans(self, array) -> None:  # pylint: disable=invalid-name
            self.assertFalse(np.isnan(array).any())


class VariationalTrainingTests(AbstractTests.TrainingTests[VariationalFullBayesianGPController]):
    """
    Basic tests for the VariationalHierarchicalHyperparameters and BayesianHyperparameters decorators.
    """

    controller_class = VariationalFullBayesianGPController


class LaplaceTrainingTests(AbstractTests.TrainingTests[LaplaceFullBayesianGPController]):
    """
    Basic tests for the LaplaceHierarchicalHyperparameters and BayesianHyperparameters decorators.
    """

    controller_class = LaplaceFullBayesianGPController

    def test_set_temperature(self) -> None:
        dataset = SyntheticDataset(rng=self.rng)
        gp = self.controller_class(
            dataset.train_x, dataset.train_y, BayesianScaledRBFKernel, dataset.train_y_std, rng=self.rng
        )
        gp.fit(10)
        plain_covariance = gp.hyperparameter_posterior.covariance_matrix.detach().cpu().numpy()
        for test_temperature in np.logspace(-3, 0, 20):
            gp.temperature = test_temperature
            new_covariance = gp.hyperparameter_posterior.covariance_matrix.detach().cpu().numpy()
            self.assertEqual(test_temperature, gp.temperature)
            np.testing.assert_array_almost_equal(plain_covariance * test_temperature, new_covariance, decimal=3)
