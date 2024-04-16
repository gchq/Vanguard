import unittest

import numpy as np
import torch
from scipy import stats

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
        np.testing.assert_array_almost_equal(ci_lower, mean - CONF_FAC*std, decimal=4)

    def test_2_dim_mean_confidence_interval(self) -> None:
        covar = torch.diag(self.std ** 2)
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
        np.testing.assert_array_almost_equal(ci_lower[:, 0], mean_1.squeeze() - CONF_FAC*std_1, decimal=3)
        np.testing.assert_array_almost_equal(ci_lower[:, 1], mean_2.squeeze() - CONF_FAC*std_2, decimal=3)

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
