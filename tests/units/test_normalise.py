"""
Tests for the NormaliseY decorator.
"""
import unittest

import torch

from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.normalise import NormaliseY
from vanguard.vanilla import GaussianGPController


@NormaliseY(ignore_methods=("__init__",))
class NormalisedGaussianGPController(GaussianGPController):
    """Test class."""
    pass


class BasicTests(unittest.TestCase):
    """
    Basic tests for the NormaliseY decorator.
    """
    @classmethod
    def setUpClass(cls) -> None:
        """Code to run before all tests."""
        cls.dataset = SyntheticDataset()

        cls.controller = NormalisedGaussianGPController(cls.dataset.train_x, cls.dataset.train_y,
                                                        ScaledRBFKernel, cls.dataset.train_y_std)

        cls.train_y_mean = cls.dataset.train_y.mean()
        cls.train_y_std = cls.dataset.train_y.std()
        cls.controller.fit(10)

    def test_pre_normalisation(self) -> None:
        """Data should probably not be normalised already!"""
        self.assertNotAlmostEqual(0, self.train_y_mean)
        self.assertNotAlmostEqual(1, self.train_y_std, delta=0.05)

    def test_normalisation(self) -> None:
        """Data should be properly normalised."""
        self.assertAlmostEqual(0, self.controller.train_y.mean().detach().item())
        self.assertAlmostEqual(1, self.controller.train_y.std().detach().item(), delta=0.05)

    def test_prediction_scaling(self) -> None:
        """Internal and external predictions should be properly scaled."""
        posterior = self.controller.posterior_over_point(self.dataset.test_x)

        internal_mean, internal_covar = posterior._tensor_prediction()
        external_mean, external_covar = posterior.prediction()

        torch.testing.assert_allclose((external_mean - self.train_y_mean) / self.train_y_std, internal_mean)
        torch.testing.assert_allclose(external_covar / self.train_y_std ** 2, internal_covar)

    def test_fuzzy_prediction_scaling(self) -> None:
        """Internal and external fuzzy predictions should be properly scaled."""
        posterior = self.controller.posterior_over_fuzzy_point(self.dataset.test_x, self.dataset.test_x_std)

        internal_mean, internal_covar = posterior._tensor_prediction()
        external_mean, external_covar = posterior.prediction()

        torch.testing.assert_allclose((external_mean - self.train_y_mean) / self.train_y_std, internal_mean)
        torch.testing.assert_allclose(external_covar / self.train_y_std ** 2, internal_covar)

    def test_confidence_interval_scaling(self) -> None:
        """Internal and external predictions should be properly scaled."""
        posterior = self.controller.posterior_over_point(self.dataset.test_x)

        internal_median, internal_upper, internal_lower = posterior._tensor_confidence_interval(0.05)
        external_median, external_upper, external_lower = posterior.confidence_interval(0.05)

        torch.testing.assert_allclose((external_median - self.train_y_mean) / self.train_y_std, internal_median)
        torch.testing.assert_allclose((external_upper - self.train_y_mean) / self.train_y_std, internal_upper)
        torch.testing.assert_allclose((external_lower - self.train_y_mean) / self.train_y_std, internal_lower)

    def test_fuzzy_confidence_interval_scaling(self) -> None:
        """Internal and external predictions should be properly scaled."""
        posterior = self.controller.posterior_over_fuzzy_point(self.dataset.test_x, self.dataset.test_x_std)

        internal_median, internal_upper, internal_lower = posterior._tensor_confidence_interval(0.05)
        external_median, external_upper, external_lower = posterior.confidence_interval(0.05)

        torch.testing.assert_allclose((external_median - self.train_y_mean) / self.train_y_std, internal_median)
        torch.testing.assert_allclose((external_upper - self.train_y_mean) / self.train_y_std, internal_upper)
        torch.testing.assert_allclose((external_lower - self.train_y_mean) / self.train_y_std, internal_lower)
