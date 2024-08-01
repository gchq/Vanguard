"""Tests for `vanguard.datasets.classification`."""

from unittest import TestCase

import numpy as np
from typing_extensions import override

from tests.cases import get_default_rng
from vanguard.datasets.classification import (
    BinaryGaussianClassificationDataset,
    BinaryStripeClassificationDataset,
    MulticlassGaussianClassificationDataset,
)


class TestBinaryStripeDataset(TestCase):
    """Tests for the `BinaryStripeClassificationDataset` class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up data shared across tests."""
        cls.num_train = 40
        cls.num_test = 20
        cls.dataset = BinaryStripeClassificationDataset(cls.num_train, cls.num_test, get_default_rng())

    def test_num_points(self) -> None:
        """Test that the dataset is generated with the correct number of points."""
        assert self.dataset.num_training_points == self.num_train
        assert self.dataset.num_testing_points == self.num_test
        assert self.dataset.num_points == self.num_train + self.num_test

    def test_inputs_are_1d(self) -> None:
        """Test that the dataset has 1-dimensional inputs."""
        assert self.dataset.num_features == 1

    def test_outputs_are_binary(self) -> None:
        """Test that the dataset has binary training targets."""
        assert all(x == 0 or x == 1 for x in self.dataset.train_y)
        assert all(x == 0 or x == 1 for x in self.dataset.test_y)


class TestMulticlassGaussianClassificationDataset(TestCase):
    """Tests for the `MulticlassGaussianClassificationDataset` class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up data shared across tests."""
        cls.num_train = 40
        cls.num_test = 20
        cls.num_classes = 6
        cls.num_features = 3

        cls.dataset = MulticlassGaussianClassificationDataset(
            cls.num_train, cls.num_test, cls.num_classes, num_features=cls.num_features, rng=get_default_rng()
        )

    def test_num_classes(self) -> None:
        """Test that there are no more than `num_classes` classes."""
        assert np.all(np.isin(self.dataset.train_y, np.arange(self.num_classes)))
        assert np.all(np.isin(self.dataset.test_y, np.arange(self.num_classes)))

    def test_num_features(self) -> None:
        """Test that `num_features` is set correctly."""
        assert self.dataset.num_features == self.num_features

    def test_one_hot_train_y(self):
        """Test that `one_hot_train_y` returns a correct one-hot encoding of the training labels."""
        assert self.dataset.one_hot_train_y.shape[:-1] == self.dataset.train_y.shape
        assert self.dataset.one_hot_train_y.shape[-1] == self.num_classes
        assert np.all(self.dataset.one_hot_train_y.argmax(axis=-1) == self.dataset.train_y)
        assert np.all(self.dataset.one_hot_train_y.sum(axis=-1) == np.ones_like(self.dataset.train_y))
        assert np.all(np.isin(self.dataset.one_hot_train_y, [0, 1]))


class TestBinaryGaussianClassificationDataset(TestMulticlassGaussianClassificationDataset):
    @override
    @classmethod
    def setUpClass(cls) -> None:
        """Set up data shared across tests."""
        cls.num_train = 40
        cls.num_test = 20
        cls.num_features = 3

        cls.dataset = BinaryGaussianClassificationDataset(
            cls.num_train, cls.num_test, num_features=cls.num_features, rng=get_default_rng()
        )

    def test_num_classes(self) -> None:
        """Test that there are no more than `num_classes` classes."""
        assert np.all(np.isin(self.dataset.train_y, [0, 1]))
        assert np.all(np.isin(self.dataset.test_y, [0, 1]))

    def test_num_features(self) -> None:
        """Test that `num_features` is set correctly."""
        assert self.dataset.num_features == self.num_features

    def test_one_hot_train_y(self):
        """Test that `one_hot_train_y` returns a correct one-hot encoding of the training labels."""
        assert np.all(self.dataset.one_hot_train_y == self.dataset.train_y.reshape((-1, 1)))
