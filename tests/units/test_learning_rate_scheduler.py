"""
Tests for ApplyLearningRateScheduler.
"""

import unittest

import torch

from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.optimise import ApplyLearningRateScheduler
from vanguard.vanilla import GaussianGPController


class BasicTests(unittest.TestCase):
    """
    Basic tests for the LearningRateFinder decorator.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.dataset = SyntheticDataset()

        num_iters = 33
        step_size = 10
        gamma = 0.9
        initial_lr = 0.2

        self.expected_lr = initial_lr * gamma ** (num_iters // step_size)

        @ApplyLearningRateScheduler(torch.optim.lr_scheduler.StepLR, step_size=step_size, gamma=gamma)
        class StepLRAdam(torch.optim.Adam):
            pass

        self.controller = GaussianGPController(
            self.dataset.train_x,
            self.dataset.train_y,
            ScaledRBFKernel,
            self.dataset.train_y_std,
            optimiser_class=StepLRAdam,
            optim_kwargs={"lr": initial_lr},
        )

        self.train_y_mean = self.dataset.train_y.mean()
        self.train_y_std = self.dataset.train_y.std()

        self.controller.fit(num_iters)

    def test_learning_rate_is_stepped(self) -> None:
        # TODO: this does look like a _lot_ of protected accesses - review this test?
        # pylint: disable=protected-access
        current_lr = self.controller._smart_optimiser._internal_optimiser._applied_scheduler.get_lr()[0]
        self.assertAlmostEqual(current_lr, self.expected_lr)
