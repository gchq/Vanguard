"""
Tests for the greedy optimisation behaviour in GreedySmartOptimiser.
"""
import unittest

import numpy as np

from vanguard.datasets.synthetic import SyntheticDataset, very_complicated_f
from vanguard.kernels import ScaledRBFKernel
from vanguard.optimise import SmartOptimiser
from vanguard.vanilla import GaussianGPController


class ParameterAgreementTests(unittest.TestCase):
    """
    Basic tests for the GreedySmartOptimiser class.

    The tests check that using the greedy optimiser gives different hyperparameters
    than when using the simpler optimiser when the learning rate is very large.
    A large learning rate is used to intentionally destabilise training so that the
    lowest loss value is almost sure to be somewhere before the final iteration.
    The tests also check that running two identical training jobs with the standard
    optimiser produces the same results, to provide assurance to the earlier tests.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Code to run before all tests."""
        cls.dataset = SyntheticDataset(functions=(very_complicated_f,), output_noise=0.9)

        cls.greedy_controller = GaussianGPController(
            cls.dataset.train_x,
            cls.dataset.train_y,
            ScaledRBFKernel,
            cls.dataset.train_y_std,
            optimiser_kwargs={"lr": 20},
        )

        cls.controller = GaussianGPController(
            cls.dataset.train_x,
            cls.dataset.train_y,
            ScaledRBFKernel,
            cls.dataset.train_y_std,
            optimiser_kwargs={"lr": 20},
            smart_optimiser_class=SmartOptimiser,
        )

        cls.controller2 = GaussianGPController(
            cls.dataset.train_x,
            cls.dataset.train_y,
            ScaledRBFKernel,
            cls.dataset.train_y_std,
            optimiser_kwargs={"lr": 20},
            smart_optimiser_class=SmartOptimiser,
        )

        cls.controller.fit(100)
        cls.controller2.fit(100)
        cls.greedy_controller.fit(100)

    def test_final_outputscales_are_different(self) -> None:
        self.assertNotAlmostEqual(
            self.controller.kernel.outputscale.item(), self.greedy_controller.kernel.outputscale.item()
        )

    def test_final_lengthscales_are_different(self) -> None:
        self.assertNotAlmostEqual(
            self.controller.kernel.base_kernel.lengthscale.item(),
            self.greedy_controller.kernel.base_kernel.lengthscale.item(),
        )

    def test_final_means_are_different(self) -> None:
        self.assertNotAlmostEqual(self.controller.mean.constant.item(), self.greedy_controller.mean.constant.item())

    def test_final_outputscales_are_same(self) -> None:
        self.assertEqual(self.controller.kernel.outputscale.item(), self.controller2.kernel.outputscale.item())

    def test_final_lengthscales_are_same(self) -> None:
        self.assertEqual(
            self.controller.kernel.base_kernel.lengthscale.item(),
            self.controller2.kernel.base_kernel.lengthscale.item(),
        )

    def test_final_means_are_same(self) -> None:
        self.assertEqual(self.controller.mean.constant.item(), self.controller2.mean.constant.item())

    def test_loss_is_best_greedy(self) -> None:
        best_loss = min(np.nan_to_num(self.greedy_controller._smart_optimiser.last_n_losses, nan=np.inf))
        used_loss = -self.greedy_controller._smart_optimiser._top_n_parameters.best().priority_value
        self.assertEqual(used_loss, best_loss)
