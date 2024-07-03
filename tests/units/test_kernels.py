"""
Tests for kernels.
"""

import unittest

import numpy as np
from numpy.testing import assert_array_less

from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import TimeSeriesKernel
from vanguard.vanilla import GaussianGPController


class BasicTests(unittest.TestCase):
    """
    Basic tests for the TimeSeriesKernel decorator.
    """

    def test_trains_time_feature_only(self) -> None:
        dataset = SyntheticDataset(rng=np.random.default_rng(1234))
        controller = GaussianGPController(dataset.train_x, dataset.train_y, TimeSeriesKernel, y_std=dataset.train_y_std)
        controller.fit(10)
        mean, _, upper = controller.posterior_over_point(dataset.test_x).confidence_interval()
        assert_array_less(upper - mean, 0.5)
