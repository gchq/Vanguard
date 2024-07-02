"""
Tests for the Multitask decorator.
"""

import unittest

import numpy as np

from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.multitask import Multitask
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference


class ErrorTests(unittest.TestCase):
    """
    Tests that the correct error messages are thrown.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.dataset = SyntheticDataset(rng=np.random.default_rng(1234))

    def test_single_task_variational(self) -> None:
        """Should throw an error."""

        @Multitask(num_tasks=1)
        @VariationalInference()
        class MultitaskController(GaussianGPController):
            pass

        with self.assertRaises(TypeError):
            MultitaskController(self.dataset.train_x, self.dataset.train_y, ScaledRBFKernel, self.dataset.train_y_std)

    def test_bad_batch_shape(self) -> None:
        """Should throw an error."""

        @Multitask(num_tasks=1)
        class MultitaskController(GaussianGPController):
            pass

        with self.assertRaises(TypeError):
            MultitaskController(
                self.dataset.train_x,
                self.dataset.train_y,
                ScaledRBFKernel,
                self.dataset.train_y_std,
                kernel_kwargs={"batch_shape": 2},
            )
