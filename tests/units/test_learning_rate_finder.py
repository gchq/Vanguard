"""
Tests for LearningRateFinder.
"""
import unittest

from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.optimise import LearningRateFinder
from vanguard.vanilla import GaussianGPController


class BasicTests(unittest.TestCase):
    """
    Basic tests for the LearningRateFinder decorator.
    """
    @classmethod
    def setUpClass(cls):
        """Code to run before all tests."""
        cls.dataset = SyntheticDataset()

        cls.controller = GaussianGPController(cls.dataset.train_x, cls.dataset.train_y,
                                              ScaledRBFKernel, cls.dataset.train_y_std)

        cls.train_y_mean = cls.dataset.train_y.mean()
        cls.train_y_std = cls.dataset.train_y.std()

        cls.lr_finder = LearningRateFinder(cls.controller)
        cls.lr_finder.find(max_iterations=10, num_divisions=25)

    def test_learning_rate_is_within_bounds(self):
        self.assertGreater(self.lr_finder.best_learning_rate, 1e-5)
        self.assertLess(self.lr_finder.best_learning_rate, 10)
