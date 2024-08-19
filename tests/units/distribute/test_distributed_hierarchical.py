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
Tests for the Distributed decorator.
"""

import unittest

import torch
from gpytorch.kernels import RBFKernel
from scipy.spatial import distance_matrix

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.distribute import Distributed, aggregators
from vanguard.hierarchical import (
    BayesianHyperparameters,
    LaplaceHierarchicalHyperparameters,
    VariationalHierarchicalHyperparameters,
)
from vanguard.vanilla import GaussianGPController


@Distributed(
    n_experts=3, aggregator_class=aggregators.GRBCMAggregator, ignore_methods=("__init__",), rng=get_default_rng()
)
@VariationalHierarchicalHyperparameters()
class DistributedVariationalHierarchicalGaussianGPController(GaussianGPController):
    """Test class."""


@Distributed(
    n_experts=3, aggregator_class=aggregators.GRBCMAggregator, ignore_methods=("__init__",), rng=get_default_rng()
)
@LaplaceHierarchicalHyperparameters()
class DistributedLaplaceHierarchicalGaussianGPController(GaussianGPController):
    """Test class."""


@BayesianHyperparameters()
class BayesianKernel(RBFKernel):
    """Test class."""


class VariationalTests(unittest.TestCase):
    """
    Tests relating to distributed variational hierarchical controllers.
    """

    def setUp(self) -> None:
        """Set up data shared between tests."""
        self.rng = get_default_rng()
        self.dataset = SyntheticDataset(n_train_points=20, n_test_points=5, rng=self.rng)

    def test_variational_distribution_is_same_on_all_experts(self) -> None:
        """
        Test that all experts share the same variational distribution.
        """
        gp = DistributedVariationalHierarchicalGaussianGPController(
            self.dataset.train_x, self.dataset.train_y, BayesianKernel, self.dataset.train_y_std, rng=self.rng
        )
        gp.fit(1)

        _ = gp.posterior_over_point(self.dataset.test_x)

        # pylint: disable=protected-access
        hyperparameter_collections = (expert.hyperparameter_collection for expert in gp._expert_controllers)
        var_dists = [collection.variational_distribution() for collection in hyperparameter_collections]

        means = torch.stack([var_dist.mean for var_dist in var_dists]).detach().cpu().numpy()
        mean_dists = distance_matrix(means, means)
        self.assertTrue((mean_dists == 0).all())

    def test_variational_distribution_from_subset_is_copied(self) -> None:
        """
        Test that experts' variation distribution matches the trained subset controller.
        """
        gp = DistributedVariationalHierarchicalGaussianGPController(
            self.dataset.train_x, self.dataset.train_y, BayesianKernel, self.dataset.train_y_std, rng=self.rng
        )
        gp.fit(1)

        _ = gp.posterior_over_point(self.dataset.test_x)

        # pylint: disable=protected-access
        hyperparameter_collections = (expert.hyperparameter_collection for expert in gp._expert_controllers)
        var_dists = [collection.variational_distribution() for collection in hyperparameter_collections]

        means = torch.stack([var_dist.mean for var_dist in var_dists]).detach().cpu().numpy()
        distribution = gp.hyperparameter_collection.variational_distribution
        fit_mean = distribution().mean.unsqueeze(dim=0).detach().cpu().numpy()
        mean_dists = distance_matrix(fit_mean, means)
        self.assertTrue((mean_dists == 0).all())


class LaplaceTests(unittest.TestCase):
    """
    Tests relating to distributed Laplace variational hierarchical controllers.
    """

    def setUp(self) -> None:
        """Set up data shared between tests."""
        self.rng = get_default_rng()
        self.dataset = SyntheticDataset(n_train_points=20, n_test_points=5, rng=self.rng)

    def test_posterior_mean_is_same_on_all_experts(self) -> None:
        """
        Test that all experts share the same mean of their variational distributions.
        """
        gp = DistributedLaplaceHierarchicalGaussianGPController(
            self.dataset.train_x, self.dataset.train_y, BayesianKernel, self.dataset.train_y_std, rng=self.rng
        )
        gp.fit(1)

        _ = gp.posterior_over_point(self.dataset.test_x)

        # pylint: disable=protected-access
        posterior_means = [expert.hyperparameter_posterior_mean for expert in gp._expert_controllers]

        means = torch.stack(posterior_means).detach().cpu().numpy()
        mean_dists = distance_matrix(means, means)
        self.assertTrue((mean_dists == 0).all())

    def test_posterior_covar_is_same_on_all_experts(self) -> None:
        """
        Test that all experts share the same covariance of their variational distributions.
        """
        dataset = SyntheticDataset(rng=self.rng)

        gp = DistributedLaplaceHierarchicalGaussianGPController(
            dataset.train_x, dataset.train_y, BayesianKernel, dataset.train_y_std, rng=self.rng
        )
        gp.fit(1)

        _ = gp.posterior_over_point(dataset.test_x)

        posterior_covariance_evals = [
            # pylint: disable=protected-access
            expert.hyperparameter_posterior_covariance[0]
            for expert in gp._expert_controllers
        ]
        posterior_covariance_evecs = [
            # pylint: disable=protected-access
            expert.hyperparameter_posterior_covariance[1]
            for expert in gp._expert_controllers
        ]

        covars_evals = torch.stack(posterior_covariance_evals).detach().cpu().numpy()
        covars_evals = covars_evals.reshape((covars_evals.shape[0], -1))
        covar_dists = distance_matrix(covars_evals, covars_evals)
        self.assertTrue((covar_dists == 0).all())

        covars_evecs = torch.stack(posterior_covariance_evecs).detach().cpu().numpy()
        covars_evecs = covars_evecs.reshape((covars_evecs.shape[0], -1))
        covar_dists = distance_matrix(covars_evecs, covars_evecs)
        self.assertTrue((covar_dists == 0).all())

    def test_posterior_mean_from_subset_is_copied(self) -> None:
        """
        Test that experts' posterior mean matches the trained subset controller.
        """
        gp = DistributedLaplaceHierarchicalGaussianGPController(
            self.dataset.train_x, self.dataset.train_y, BayesianKernel, self.dataset.train_y_std, rng=self.rng
        )
        gp.fit(1)

        _ = gp.posterior_over_point(self.dataset.test_x)

        # pylint: disable=protected-access
        posterior_means = [expert.hyperparameter_posterior_mean for expert in gp._expert_controllers]
        means = torch.stack(posterior_means).detach().cpu().numpy()
        fit_mean = gp.hyperparameter_posterior_mean.unsqueeze(dim=0).detach().cpu().numpy()
        mean_dists = distance_matrix(fit_mean, means)
        self.assertTrue((mean_dists == 0).all())

    def test_posterior_covariance_from_subset_is_copied(self) -> None:
        """
        Test that experts' posterior covariance matches the trained subset controller.
        """
        gp = DistributedLaplaceHierarchicalGaussianGPController(
            self.dataset.train_x, self.dataset.train_y, BayesianKernel, self.dataset.train_y_std, rng=self.rng
        )
        gp.fit(1)

        _ = gp.posterior_over_point(self.dataset.test_x)

        posterior_covariance_evals = [
            # pylint: disable=protected-access
            expert.hyperparameter_posterior_covariance[0]
            for expert in gp._expert_controllers
        ]
        posterior_covariance_evecs = [
            # pylint: disable=protected-access
            expert.hyperparameter_posterior_covariance[1]
            for expert in gp._expert_controllers
        ]
        covars_evals = torch.stack(posterior_covariance_evals).detach().cpu().numpy()
        covars_evals = covars_evals.reshape((covars_evals.shape[0], -1))
        covars_evecs = torch.stack(posterior_covariance_evecs).detach().cpu().numpy()
        covars_evecs = covars_evecs.reshape((covars_evecs.shape[0], -1))

        fit_covar_evals = gp.hyperparameter_posterior_covariance[0].detach().cpu().numpy().reshape((1, -1))
        fit_covar_evecs = gp.hyperparameter_posterior_covariance[1].detach().cpu().numpy().reshape((1, -1))
        covar_dists = distance_matrix(fit_covar_evals, covars_evals)
        self.assertTrue((covar_dists == 0).all())
        covar_dists = distance_matrix(fit_covar_evecs, covars_evecs)
        self.assertTrue((covar_dists == 0).all())
