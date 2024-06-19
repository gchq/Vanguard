"""
Tests for the Posterior class.
"""

import unittest
from unittest.mock import Mock

import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from scipy import stats
from scipy.stats import multivariate_normal

from vanguard.base.posteriors import Posterior

CONF_FAC = stats.norm.ppf(0.975)


class BasicTests(unittest.TestCase):
    """
    Basic tests for the Posterior class.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.mean = torch.as_tensor([1, 2, 3, 4, 5])
        self.std = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    def test_1_dim_mean_confidence_interval(self) -> None:
        covar = torch.diag(self.std**2)
        posterior = Posterior.from_mean_and_covariance(self.mean, covar)
        _, ci_lower, _ = posterior.confidence_interval(0.05)
        mean, std = self.mean.detach().cpu().numpy(), self.std.detach().cpu().numpy()
        np.testing.assert_array_almost_equal(ci_lower, mean - CONF_FAC * std, decimal=4)

    def test_2_dim_mean_confidence_interval(self) -> None:
        covar = torch.diag(self.std**2)
        posterior = Posterior.from_mean_and_covariance(self.mean.unsqueeze(dim=-1), covar)
        _, ci_lower, _ = posterior.confidence_interval(0.05)
        mean, std = self.mean.detach().cpu().numpy(), self.std.detach().cpu().numpy()
        np.testing.assert_array_almost_equal(ci_lower, mean - CONF_FAC * std, decimal=4)

    def test_2_task_confidence_interval(self) -> None:
        mean1, std1 = self.mean, self.std
        mean2, std2 = -self.mean, self.std * 0.15
        covar = torch.diag(torch.cat([std1, std2]) ** 2)
        posterior = Posterior.from_mean_and_covariance(torch.stack([mean1, mean2], -1), covar)
        _, ci_lower, _ = posterior.confidence_interval(0.05)
        mean_1, std_1 = mean1.detach().cpu().numpy(), std1.detach().cpu().numpy()
        mean_2, std_2 = mean2.detach().cpu().numpy(), std2.detach().cpu().numpy()
        np.testing.assert_array_almost_equal(ci_lower[:, 0], mean_1.squeeze() - CONF_FAC * std_1, decimal=3)
        np.testing.assert_array_almost_equal(ci_lower[:, 1], mean_2.squeeze() - CONF_FAC * std_2, decimal=3)

    def test_1_dim_mean_log_probability_size(self) -> None:
        covar = torch.diag(self.std**2)
        posterior = Posterior.from_mean_and_covariance(self.mean, covar)
        log_prob = posterior.log_probability(torch.randn(*self.mean.shape))
        self.assertLess(log_prob, 0)

    def test_1_dim_mean_log_probability_order(self) -> None:
        covar = torch.diag(self.std**2)
        posterior = Posterior.from_mean_and_covariance(self.mean, covar)
        log_prob_rand = posterior.log_probability(torch.randn(*self.mean.shape))
        log_prob_centre = posterior.log_probability(self.mean)
        self.assertLess(log_prob_rand, log_prob_centre)

    def test_sample(self) -> None:
        mock_distribution = Mock()
        del mock_distribution.covariance_matrix  # no covariance matrix, so no jitter is added
        mock_distribution.rsample.side_effect = torch.ones

        posterior = Posterior(mock_distribution)

        for size in [0, 1, 10]:
            with self.subTest(size=size):
                np.testing.assert_array_equal(np.ones(size), posterior.sample(size))

    def test_log_probability(self):
        """
        Test that the log_probability method works as expected when a two-dimensional sample is passed in.

        ...and when the collection consists of identical posteriors.
        """
        posterior = Posterior(MultivariateNormal(torch.zeros((2,)), torch.eye(2)))

        distribution = multivariate_normal(np.zeros((2,)), np.eye(2), seed=1234)

        # for 10 random points, check that the log-pdf calculated by the collection is the same as that for the
        # single distribution
        for _ in range(10):
            points = distribution.rvs([1, 5])
            with self.subTest(points=points):
                expected_value = np.sum(distribution.logpdf(points), axis=0)
                return_value = posterior.log_probability(points)
                self.assertAlmostEqual(expected_value, return_value, delta=1e-4)
