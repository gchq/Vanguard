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

CONF_INTERVAL_SIZE = 0.05
CONF_FAC = stats.norm.ppf(1 - (CONF_INTERVAL_SIZE / 2))


class BasicTests(unittest.TestCase):
    """
    Basic tests for the Posterior class.
    """

    def setUp(self) -> None:
        """Set up data shared across tests."""
        self.mean = torch.as_tensor([1, 2, 3, 4, 5])
        self.std = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    def test_mean_confidence_interval(self) -> None:
        """
        Test that the confidence interval is calculated correctly.

        We test both the one-dimensional case and the two-dimensional case. (The two-dimensional case only checks the
        case with a single task, though.)
        """
        covar = torch.diag(self.std**2)
        for posterior_mean, msg in [
            (self.mean, "1-dimensional"),
            (self.mean.unsqueeze(dim=-1), "2-dimensional"),
        ]:
            with self.subTest(msg):
                posterior = Posterior.from_mean_and_covariance(posterior_mean, covar)
                ci_median, ci_lower, ci_upper = posterior.confidence_interval(CONF_INTERVAL_SIZE)
                mean, std = self.mean.detach().cpu().numpy(), self.std.detach().cpu().numpy()

                np.testing.assert_array_almost_equal(ci_lower, mean - CONF_FAC * std, decimal=4)
                np.testing.assert_array_almost_equal(ci_median, mean, decimal=4)
                np.testing.assert_array_almost_equal(ci_upper, mean + CONF_FAC * std, decimal=4)

    def test_2_task_confidence_interval(self) -> None:
        """
        Test that the confidence interval is calculated correctly in the 2-task case.
        """
        mean1, std1 = self.mean, self.std
        mean2, std2 = -self.mean, self.std * 0.15
        covar = torch.diag(torch.cat([std1, std2]) ** 2)
        posterior = Posterior.from_mean_and_covariance(torch.stack([mean1, mean2], -1), covar)
        ci_median, ci_lower, ci_upper = posterior.confidence_interval(0.05)
        mean_1, std_1 = mean1.detach().cpu().numpy(), std1.detach().cpu().numpy()
        mean_2, std_2 = mean2.detach().cpu().numpy(), std2.detach().cpu().numpy()

        # assert results are as expected for task 1
        np.testing.assert_array_almost_equal(ci_lower[:, 0], mean_1.squeeze() - CONF_FAC * std_1, decimal=3)
        np.testing.assert_array_almost_equal(ci_median[:, 0], mean_1.squeeze(), decimal=3)
        np.testing.assert_array_almost_equal(ci_upper[:, 0], mean_1.squeeze() + CONF_FAC * std_1, decimal=3)
        # assert results are as expected for task 2
        np.testing.assert_array_almost_equal(ci_lower[:, 1], mean_2.squeeze() - CONF_FAC * std_2, decimal=3)
        np.testing.assert_array_almost_equal(ci_median[:, 1], mean_2.squeeze(), decimal=3)
        np.testing.assert_array_almost_equal(ci_upper[:, 1], mean_2.squeeze() + CONF_FAC * std_2, decimal=3)

    def test_1_dim_mean_log_probability_size(self) -> None:
        """
        Test that Posterior's log_probability is within sane bounds in the 1-dimensional case.

        We only check that the log_probability is negative (and hence that the probability is less than 1).
        """
        covar = torch.diag(self.std**2)
        posterior = Posterior.from_mean_and_covariance(self.mean, covar)
        log_prob = posterior.log_probability(torch.randn(*self.mean.shape))
        self.assertLess(log_prob, 0)

    def test_1_dim_mean_log_probability_order(self) -> None:
        """
        Test that Posterior's log_probability has a maximum at its mean.

        We do this by checking that the log_probability is lower at a number of randomly selected points than at the
        mean.
        """
        covar = torch.diag(self.std**2)
        posterior = Posterior.from_mean_and_covariance(self.mean, covar)

        generator = torch.Generator().manual_seed(1234)  # seeded for consistency

        for _ in range(10):
            random_point = torch.randn(*self.mean.shape, generator=generator)
            with self.subTest(point=random_point):
                log_prob_rand = posterior.log_probability(random_point)
                log_prob_centre = posterior.log_probability(self.mean)
                self.assertLess(log_prob_rand, log_prob_centre)

    def test_sample(self) -> None:
        """Test that the sample() function simply returns a sample from the internal distribution."""
        # Set up a mock distribution
        mock_distribution = Mock()
        del mock_distribution.covariance_matrix  # no covariance matrix, so no jitter is added

        # Define a function to generate and record random samples, and replace our mock distribution's `rsample`
        # method with it
        samples = []
        generator = torch.Generator().manual_seed(1234)

        def rsample(sample_shape: torch.Size):
            """Dummy sample function that generates and records random samples."""
            sample = torch.randn(*sample_shape, generator=generator)
            samples.append(sample)
            return sample

        mock_distribution.rsample.side_effect = rsample

        posterior = Posterior(mock_distribution)

        # Test that sample() just calls our dummy function as expected
        for size in range(10):
            with self.subTest(size=size):
                sample = posterior.sample(size)
                mock_sample = samples[-1]
                np.testing.assert_array_equal(mock_sample, sample)

    def test_log_probability_2d(self):
        """
        Test that the log_probability method works as expected when a two-dimensional sample is passed in.
        """
        posterior = Posterior(MultivariateNormal(torch.zeros((2,)), torch.eye(2)))

        distribution = multivariate_normal(np.zeros((2,)), np.eye(2), seed=1234)

        for _ in range(10):
            points = distribution.rvs([1, 5])
            with self.subTest(points=points):
                expected_value = np.sum(distribution.logpdf(points), axis=0)
                return_value = posterior.log_probability(points)
                self.assertAlmostEqual(expected_value, return_value, delta=1e-4)
