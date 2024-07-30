"""Tests for `vanguard.datasets.synthetic`."""

from unittest import TestCase

import numpy as np

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import (
    HeteroskedasticSyntheticDataset,
    HigherRankSyntheticDataset,
    MultidimensionalSyntheticDataset,
    SyntheticDataset,
    simple_f,
)


class TestSynthetic(TestCase):
    """Tests for the `SyntheticDataset` class."""

    @classmethod
    def setUpClass(cls):
        """Set up data shared between tests."""
        cls.num_train = 50
        cls.num_test = 20
        cls.output_noise = 0.1
        rng = get_default_rng()
        cls.default_dataset = SyntheticDataset(
            n_train_points=cls.num_train, n_test_points=cls.num_test, output_noise=cls.output_noise, rng=rng
        )

        cls.multi_function_num_functions = 3
        cls.multi_function_dataset = SyntheticDataset(
            functions=[simple_f for _ in range(cls.multi_function_num_functions)],
            n_train_points=cls.num_train,
            n_test_points=cls.num_test,
            output_noise=cls.output_noise,
            rng=rng,
        )

    def test_num_points(self):
        """Test that the number of points generated is correct."""
        assert self.default_dataset.num_training_points == self.num_train
        assert self.default_dataset.num_testing_points == self.num_test

    def test_y_standard_deviation(self):
        """Test that the y standard deviation is set to whatever was entered for the `output_noise`."""
        assert self.default_dataset.train_y_std == self.output_noise
        assert self.default_dataset.test_y_std == self.output_noise

    def test_num_features(self):
        """Test that the default dataset has exactly 1 feature."""
        assert self.default_dataset.num_features == 1

    def test_multiple_functions_num_features(self):
        """Test that when multiple functions are passed, the dataset still has only 1 feature."""
        assert self.multi_function_dataset.num_features == 1

    def test_multiple_functions_num_output_dimensions(self):
        """Test that when multiple functions are passed, the output is multidimensional."""
        assert self.multi_function_dataset.train_y.shape == (self.num_train, self.multi_function_num_functions)
        assert self.multi_function_dataset.test_y.shape == (self.num_test, self.multi_function_num_functions)


class TestMultidimensionalSynthetic(TestCase):
    """Tests for `MultidimensionalSyntheticDataset` class."""

    @classmethod
    def setUpClass(cls):
        """Set up data shared between tests."""
        cls.num_train = 50
        cls.num_test = 20
        cls.output_noise = 0.1
        cls.num_functions = 3
        rng = get_default_rng()
        cls.dataset = MultidimensionalSyntheticDataset(
            functions=[simple_f for _ in range(cls.num_functions)],
            n_train_points=cls.num_train,
            n_test_points=cls.num_test,
            output_noise=cls.output_noise,
            rng=rng,
        )

    def test_num_points(self):
        """Test that the number of points generated is correct."""
        assert self.dataset.num_training_points == self.num_train
        assert self.dataset.num_testing_points == self.num_test

    def test_y_standard_deviation(self):
        """Test that the y standard deviation is set to whatever was entered for the `output_noise`."""
        assert self.dataset.train_y_std == self.output_noise
        assert self.dataset.test_y_std == self.output_noise

    def test_num_features(self):
        """Test that the default dataset has a feature per function passed."""
        assert self.dataset.num_features == self.num_functions

    def test_output_one_dimensional(self):
        """Test that the output is 1-dimensional."""
        assert self.dataset.train_y.ndim == 1


class TestHeteroskedasticSyntheticDataset(TestCase):
    def test_y_std_is_clamped(self):
        """
        Test that the `(train|test)_y_std` values are all clamped to be non-negative.

        This is tested by setting `output_noise_mean` quite low, and `output_noise_std` high.
        """
        dataset = HeteroskedasticSyntheticDataset(output_noise_mean=0.1, output_noise_std=1)
        assert np.all(dataset.train_y_std >= 0)
        assert np.all(dataset.test_y_std >= 0)


class TestHigherRankSyntheticDataset(TestCase):
    def test_x_is_2x2(self):
        """Test that the x values are 2x2 matrices."""
        num_train = 20
        num_test = 10
        dataset = HigherRankSyntheticDataset(n_train_points=num_train, n_test_points=num_test)

        assert dataset.train_x.shape == (num_train, 2, 2)
        assert dataset.test_x.shape == (num_test, 2, 2)
