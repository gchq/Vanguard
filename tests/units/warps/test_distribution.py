"""
Test the behaviour of the WarpedGaussian distribution.
"""

import unittest

import numpy as np
import torch

from vanguard.warps import MultitaskWarpFunction, warpfunctions
from vanguard.warps.distribution import WarpedGaussian


class DistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.warp = MultitaskWarpFunction(
            warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(b=-0.99),
            warpfunctions.AffineWarpFunction(),
            warpfunctions.AffineWarpFunction(),
        ).freeze()
        self.mean = torch.as_tensor([0, 1, -1]).float()
        self.scale = torch.as_tensor([0.1, 0.2, 0.3]).float()
        self.distribution = WarpedGaussian(self.warp, loc=self.mean, scale=self.scale)
        self.valid_data = torch.as_tensor([[2, 3, 4], [10, 11, 12]])
        self.invalid_data = torch.as_tensor([[0.9, 2, 3]])
        self.edge_data = torch.as_tensor([[0.99, 2, 3]])
        self.samples = self.distribution.sample((100000,)).detach()

    def test_log_prob_error_away_from_support(self) -> None:
        with self.assertRaises(ValueError):
            self.distribution.log_prob(self.invalid_data).sum().item()

    def test_log_prob_blows_up_at_edge_of_support(self) -> None:
        log_prob = self.distribution.log_prob(self.edge_data).sum().item()
        self.assertTrue(np.isnan(log_prob))

    def test_log_prob_finite_in_support(self) -> None:
        log_prob = self.distribution.log_prob(self.valid_data).sum().item()
        self.assertFalse(np.isnan(log_prob))

    def test_sample_shape_is_correct(self) -> None:
        samples = self.distribution.sample((100,))
        self.assertEqual(samples.shape, (100, 3))

    def test_gaussian_parameters_can_be_recovered(self) -> None:
        fit_distribution = WarpedGaussian.from_data(self.warp, self.samples, n_iterations=10)
        fit_distribution_loc = fit_distribution.loc.detach().cpu().numpy()
        original_distribution_loc = self.distribution.loc.detach().cpu().numpy()
        np.testing.assert_array_almost_equal(fit_distribution_loc[0], original_distribution_loc[0], decimal=3)
        np.testing.assert_array_almost_equal(fit_distribution_loc[1:], original_distribution_loc[1:], decimal=1)

    def test_fit_warp_is_close_to_optimal_in_log_prob(self) -> None:
        candidate_warp = MultitaskWarpFunction(
            warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(),
            warpfunctions.AffineWarpFunction().freeze(),
            warpfunctions.AffineWarpFunction().freeze(),
        )
        fit_distribution = WarpedGaussian.from_data(candidate_warp, self.samples, n_iterations=10)
        fit_log_prob = fit_distribution.log_prob(self.samples).mean().item()
        true_log_prob = self.distribution.log_prob(self.samples).mean().item()
        self.assertAlmostEqual(fit_log_prob, true_log_prob, places=2)

    def test_fit_warp_is_much_better_than_gaussian_approximation(self) -> None:
        candidate_warp = MultitaskWarpFunction(
            warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(),
            warpfunctions.AffineWarpFunction().freeze(),
            warpfunctions.AffineWarpFunction().freeze(),
        )
        fit_distribution = WarpedGaussian.from_data(candidate_warp, self.samples, n_iterations=10)
        fit_log_prob = fit_distribution.log_prob(self.samples).mean().item()

        gaussian_approximation = torch.distributions.Normal(loc=self.samples.mean(dim=0), scale=self.samples.std(dim=0))
        gaussian_approximation_log_prob = gaussian_approximation.log_prob(self.samples).mean().item()
        self.assertGreater(fit_log_prob, -np.abs(gaussian_approximation_log_prob) * 10)
