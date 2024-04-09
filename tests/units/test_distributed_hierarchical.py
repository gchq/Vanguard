"""
Tests for the Distributed decorator.
"""
import unittest

import torch
from gpytorch.kernels import RBFKernel
from scipy.spatial import distance_matrix

from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.distribute import Distributed, aggregators
from vanguard.hierarchical import (BayesianHyperparameters,
                                   LaplaceHierarchicalHyperparameters,
                                   VariationalHierarchicalHyperparameters)
from vanguard.vanilla import GaussianGPController


@Distributed(n_experts=10, aggregator_class=aggregators.GRBCMAggregator, ignore_methods=("__init__",))
@VariationalHierarchicalHyperparameters()
class DistributedVariationalHierarchicalGaussianGPController(GaussianGPController):
    """Test class."""
    pass


@Distributed(n_experts=10, aggregator_class=aggregators.GRBCMAggregator, ignore_methods=("__init__",))
@LaplaceHierarchicalHyperparameters()
class DistributedLaplaceHierarchicalGaussianGPController(GaussianGPController):
    """Test class."""
    pass


@BayesianHyperparameters()
class BayesianKernel(RBFKernel):
    pass


class VariationalTests(unittest.TestCase):
    """
    Some tests.
    """
    def test_variational_distribution_is_same_on_all_experts(self):
        """All experts should share variational distribution."""
        dataset = SyntheticDataset()

        gp = DistributedVariationalHierarchicalGaussianGPController(dataset.train_x, dataset.train_y,
                                                                    BayesianKernel, dataset.train_y_std)
        gp.fit(10)

        _ = gp.posterior_over_point(dataset.test_x)

        hyperparameter_collections = (expert.hyperparameter_collection for expert in gp._expert_controllers)
        var_dists = [collection.variational_distribution() for collection in hyperparameter_collections]

        means = torch.stack([var_dist.mean for var_dist in var_dists]).detach().cpu().numpy()
        mean_dists = distance_matrix(means, means)
        self.assertTrue((mean_dists == 0).all())

    def test_variational_distribution_from_subset_is_copied(self):
        """Experts' variation distribution should match the trained subset controller."""
        dataset = SyntheticDataset()

        gp = DistributedVariationalHierarchicalGaussianGPController(dataset.train_x, dataset.train_y,
                                                                    BayesianKernel, dataset.train_y_std)
        gp.fit(10)

        _ = gp.posterior_over_point(dataset.test_x)

        hyperparameter_collections = (expert.hyperparameter_collection for expert in gp._expert_controllers)
        var_dists = [collection.variational_distribution() for collection in hyperparameter_collections]

        means = torch.stack([var_dist.mean for var_dist in var_dists]).detach().cpu().numpy()
        distribution = gp.hyperparameter_collection.variational_distribution
        fit_mean = distribution().mean.unsqueeze(dim=0).detach().cpu().numpy()
        mean_dists = distance_matrix(fit_mean, means)
        self.assertTrue((mean_dists == 0).all())


class LaplaceTests(unittest.TestCase):
    """
    Some tests.
    """
    def test_posterior_mean_is_same_on_all_experts(self):
        """All experts should share variational distribution."""
        dataset = SyntheticDataset()

        gp = DistributedLaplaceHierarchicalGaussianGPController(dataset.train_x, dataset.train_y,
                                                                BayesianKernel, dataset.train_y_std)
        gp.fit(10)

        _ = gp.posterior_over_point(dataset.test_x)

        posterior_means = [expert.hyperparameter_posterior_mean for expert in gp._expert_controllers]

        means = torch.stack(posterior_means).detach().cpu().numpy()
        mean_dists = distance_matrix(means, means)
        self.assertTrue((mean_dists == 0).all())

    def test_posterior_covar_is_same_on_all_experts(self):
        """All experts should share variational distribution."""
        dataset = SyntheticDataset()

        gp = DistributedLaplaceHierarchicalGaussianGPController(dataset.train_x, dataset.train_y,
                                                                BayesianKernel, dataset.train_y_std)
        gp.fit(10)

        _ = gp.posterior_over_point(dataset.test_x)

        posterior_covariance_evals = [expert.hyperparameter_posterior_covariance[0]
                                      for expert in gp._expert_controllers]
        posterior_covariance_evecs = [expert.hyperparameter_posterior_covariance[1]
                                      for expert in gp._expert_controllers]

        covars_evals = torch.stack(posterior_covariance_evals).detach().cpu().numpy()
        covars_evals = covars_evals.reshape((covars_evals.shape[0], -1))
        covar_dists = distance_matrix(covars_evals, covars_evals)
        self.assertTrue((covar_dists == 0).all())

        covars_evecs = torch.stack(posterior_covariance_evecs).detach().cpu().numpy()
        covars_evecs = covars_evecs.reshape((covars_evecs.shape[0], -1))
        covar_dists = distance_matrix(covars_evecs, covars_evecs)
        self.assertTrue((covar_dists == 0).all())

    def test_posterior_mean_from_subset_is_copied(self):
        """Experts' posterior mean should match the trained subset controller."""
        dataset = SyntheticDataset()

        gp = DistributedLaplaceHierarchicalGaussianGPController(dataset.train_x, dataset.train_y,
                                                                BayesianKernel, dataset.train_y_std)
        gp.fit(10)

        _ = gp.posterior_over_point(dataset.test_x)

        posterior_means = [expert.hyperparameter_posterior_mean for expert in gp._expert_controllers]
        means = torch.stack(posterior_means).detach().cpu().numpy()
        fit_mean = gp.hyperparameter_posterior_mean.unsqueeze(dim=0).detach().cpu().numpy()
        mean_dists = distance_matrix(fit_mean, means)
        self.assertTrue((mean_dists == 0).all())

    def test_posterior_covariance_from_subset_is_copied(self):
        """Experts' posterior covariance should match the trained subset controller."""
        dataset = SyntheticDataset()

        gp = DistributedLaplaceHierarchicalGaussianGPController(dataset.train_x, dataset.train_y,
                                                                BayesianKernel, dataset.train_y_std)
        gp.fit(10)

        _ = gp.posterior_over_point(dataset.test_x)

        posterior_covariance_evals = [expert.hyperparameter_posterior_covariance[0]
                                      for expert in gp._expert_controllers]
        posterior_covariance_evecs = [expert.hyperparameter_posterior_covariance[1]
                                      for expert in gp._expert_controllers]
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
