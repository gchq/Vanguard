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
Tests for the LaplaceHierarchicalHyperparameters decorators.
"""

import unittest
from typing import Any

import gpytorch.distributions.multivariate_normal
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.hierarchical import (
    BayesianHyperparameters,
    LaplaceHierarchicalHyperparameters,
)
from vanguard.vanilla import GaussianGPController

N_MC_SAMPLES = 13


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


class LaplaceTrainingTests(unittest.TestCase):
    """
    Basic tests for the LaplaceHierarchicalHyperparameters and BayesianHyperparameters decorators.
    """

    controller_class = LaplaceFullBayesianGPController

    def setUp(self) -> None:
        """
        Define objects shared across tests
        """
        self.synthetic_dataset = SyntheticDataset(rng=get_default_rng())
        gp_synthetic_dataset = self.controller_class(
            self.synthetic_dataset.train_x,
            self.synthetic_dataset.train_y,
            BayesianScaledRBFKernel,
            self.synthetic_dataset.train_y_std,
            rng=get_default_rng(),
        )
        gp_synthetic_dataset.fit(10)
        self.gp_synthetic_dataset = gp_synthetic_dataset

    def test_set_temperature(self) -> None:
        """
        Test that the temperature parameter can be set on a Bayesian kernel and Bayesian controller.
        """
        plain_covariance = self.gp_synthetic_dataset.hyperparameter_posterior.covariance_matrix.detach().cpu().numpy()
        for test_temperature in np.logspace(-3, 0, 20):
            self.gp_synthetic_dataset.temperature = test_temperature
            new_covariance = self.gp_synthetic_dataset.hyperparameter_posterior.covariance_matrix.detach().cpu().numpy()
            self.assertEqual(test_temperature, self.gp_synthetic_dataset.temperature)
            np.testing.assert_array_almost_equal(plain_covariance * test_temperature, new_covariance, decimal=3)

    def test_len_redefine(self) -> None:
        """
        Test the redefinition of the length attribute on hyperparameter collections.
        """
        # We expect 3 Bayesian hyperparameters - the mean and then the kernel output scale and lengthscale
        self.assertEqual(len(self.gp_synthetic_dataset.hyperparameter_collection), 3)

    def test_log_prior_term(self) -> None:
        """
        Test the `log_prior_term` method returns expected results.
        """
        # The exact value of the log prior term is hard to verify, so we instead check the type and that it is not nan
        log_prior_result = self.gp_synthetic_dataset.hyperparameter_collection.log_prior_term()
        self.assertTrue(torch.is_tensor(log_prior_result))
        log_prior_result = log_prior_result.detach().cpu().numpy()
        self.assertFalse(np.isnan(log_prior_result))


class GeneratorTests(unittest.TestCase):
    """
    Tests for the generators defined within Bayesian GP controllers.

    Note that we have to swap the generator in and out of eval mode, so these tests are intentionally separated
    from the general tests for the Bayesian controllers.
    """

    controller_class = LaplaceFullBayesianGPController

    def setUp(self) -> None:
        """
        Define objects shared across tests.
        """
        self.synthetic_dataset = SyntheticDataset(rng=get_default_rng())
        gp_synthetic_dataset = self.controller_class(
            self.synthetic_dataset.train_x,
            self.synthetic_dataset.train_y,
            BayesianScaledRBFKernel,
            self.synthetic_dataset.train_y_std,
            rng=get_default_rng(),
        )
        gp_synthetic_dataset.fit(10)
        gp_synthetic_dataset.set_to_evaluation_mode()
        self.gp_synthetic_dataset = gp_synthetic_dataset

    def test_infinite_fuzzy_posterior_samples(self) -> None:
        """
        Test the generator provided by `_infinite_fuzzy_posterior_samples` gives expected outputs.

        We cannot test if a generator provides an infinite number of samples, so instead we verify that
        the first few outputs are of the expected type.
        """
        # pylint: disable=protected-access
        sample_generator = LaplaceHierarchicalHyperparameters()._infinite_fuzzy_posterior_samples(
            self.gp_synthetic_dataset, self.synthetic_dataset.test_x[0, :], 0.1
        )
        # pylint: enable=protected-access

        for _ in range(5):
            current_sample = next(sample_generator)

            # We expect the sample to be a MultivariateNormal distribution with a univariate mean since we only passed
            # a single test point to predict
            self.assertTrue(isinstance(current_sample, gpytorch.distributions.multivariate_normal.MultivariateNormal))
            self.assertEqual(current_sample.loc.shape, torch.Size([1]))

    def test_infinite_likelihood_samples(self) -> None:
        """
        Test the generator provided by `_infinite_likelihood_samples` gives expected outputs.

        We cannot test if a generator provides an infinite number of samples, so instead we verify that
        the first few outputs are of the expected type.
        """
        # pylint: disable=protected-access
        sample_generator = LaplaceHierarchicalHyperparameters()._infinite_likelihood_samples(
            self.gp_synthetic_dataset, self.synthetic_dataset.test_x[0, :]
        )
        # pylint: enable=protected-access

        for _ in range(5):
            current_sample = next(sample_generator)

            # We expect the sample to be a MultivariateNormal distribution with a univariate mean since we only passed
            # a single test point to predict
            self.assertTrue(isinstance(current_sample, gpytorch.distributions.multivariate_normal.MultivariateNormal))
            self.assertEqual(current_sample.loc.shape, torch.Size([1]))

    def test_infinite_fuzzy_likelihood_samples(self) -> None:
        """
        Test the generator provided by `_infinite_fuzzy_likelihood_samples` gives expected outputs.

        We cannot test if a generator provides an infinite number of samples, so instead we verify that
        the first few outputs are of the expected type.
        """
        # pylint: disable=protected-access
        sample_generator = LaplaceHierarchicalHyperparameters()._infinite_fuzzy_likelihood_samples(
            self.gp_synthetic_dataset, self.synthetic_dataset.test_x[0, :], 0.1
        )
        # pylint: enable=protected-access

        for _ in range(5):
            current_sample = next(sample_generator)

            # We expect the sample to be a MultivariateNormal distribution with a univariate mean since we only passed
            # a single test point to predict
            self.assertTrue(isinstance(current_sample, gpytorch.distributions.multivariate_normal.MultivariateNormal))
            self.assertEqual(current_sample.loc.shape, torch.Size([1]))
