"""
Tests for the SetWarp decorator.
"""
import unittest

import torch

from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.warps import SetWarp, warpfunctions


@SetWarp(warpfunctions.AffineWarpFunction(a=3, b=-1) @ warpfunctions.BoxCoxWarpFunction(0.2), ignore_all=True)
class WarpedGaussianGPController(GaussianGPController):
    """Test class."""

    pass


class BasicTests(unittest.TestCase):
    """
    Basic tests for the SetWarp decorator.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Code to run before all tests."""
        cls.dataset = SyntheticDataset()
        cls.controller = WarpedGaussianGPController(
            cls.dataset.train_x, cls.dataset.train_y, ScaledRBFKernel, cls.dataset.train_y_std
        )
        cls.controller.fit(10)

    def test_prediction_error(self) -> None:
        """Should throw a TypeError."""
        posterior = self.controller.posterior_over_point(self.dataset.test_x)
        try:
            posterior._tensor_prediction()
        except TypeError as error:
            self.fail(f"Should not have thrown {type(error)}")

        with self.assertRaises(TypeError):
            posterior.prediction()

    def test_fuzzy_prediction_error(self) -> None:
        """Should throw a TypeError."""
        posterior = self.controller.posterior_over_fuzzy_point(self.dataset.test_x, self.dataset.test_x_std)
        try:
            posterior._tensor_prediction()
        except TypeError as error:
            self.fail(f"Should not have thrown {type(error)}")

        with self.assertRaises(TypeError):
            posterior.prediction()

    def test_confidence_interval_scaling(self) -> None:
        """Internal and external predictions should be properly scaled."""
        posterior = self.controller.posterior_over_point(self.dataset.test_x)

        internal_median, internal_upper, internal_lower = posterior._tensor_confidence_interval(0.05)
        external_median, external_upper, external_lower = posterior.confidence_interval(0.05)

        warped_external_median = self.controller.warp(
            torch.as_tensor(external_median, dtype=torch.float32).reshape(-1, 1)
        )
        warped_external_upper = self.controller.warp(
            torch.as_tensor(external_upper, dtype=torch.float32).reshape(-1, 1)
        )
        warped_external_lower = self.controller.warp(
            torch.as_tensor(external_lower, dtype=torch.float32).reshape(-1, 1)
        )

        torch.testing.assert_allclose(warped_external_median, internal_median.reshape(-1, 1))
        torch.testing.assert_allclose(warped_external_lower, internal_lower.reshape(-1, 1))
        torch.testing.assert_allclose(warped_external_upper, internal_upper.reshape(-1, 1))

    def test_fuzzy_confidence_interval_scaling(self) -> None:
        """Internal and external predictions should be properly scaled."""
        posterior = self.controller.posterior_over_fuzzy_point(self.dataset.test_x, self.dataset.test_x_std)

        internal_median, internal_upper, internal_lower = posterior._tensor_confidence_interval(0.05)
        external_median, external_upper, external_lower = posterior.confidence_interval(0.05)

        warped_external_median = self.controller.warp(
            torch.as_tensor(external_median, dtype=torch.float32).reshape(-1, 1)
        )
        warped_external_upper = self.controller.warp(
            torch.as_tensor(external_upper, dtype=torch.float32).reshape(-1, 1)
        )
        warped_external_lower = self.controller.warp(
            torch.as_tensor(external_lower, dtype=torch.float32).reshape(-1, 1)
        )

        torch.testing.assert_allclose(warped_external_median, internal_median.reshape(-1, 1))
        torch.testing.assert_allclose(warped_external_upper, internal_upper.reshape(-1, 1))
        torch.testing.assert_allclose(warped_external_lower, internal_lower.reshape(-1, 1))
