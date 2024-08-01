"""Tests for the classes in basedataset.py."""

from unittest import TestCase

import numpy as np

from vanguard.datasets import Dataset, EmptyDataset


class TestBaseDataset(TestCase):
    """Tests for the base `Dataset` class."""

    @classmethod
    def setUpClass(cls):
        """Set up data shared across tests."""
        cls.num_test_points = 10
        cls.num_train_points = 20
        cls.num_features = 3

        cls.dataset = Dataset(
            train_x=np.zeros((cls.num_train_points, cls.num_features)),
            train_x_std=np.ones(cls.num_train_points),
            train_y=np.zeros(cls.num_train_points),
            train_y_std=np.ones(cls.num_train_points),
            test_x=np.zeros((cls.num_test_points, cls.num_features)),
            test_x_std=np.ones(cls.num_test_points),
            test_y=np.zeros(cls.num_test_points),
            test_y_std=np.ones(cls.num_test_points),
            significance=0.1,
        )

    def test_num_features(self) -> None:
        """Test that the number of features is returned correctly."""
        assert self.dataset.num_features == self.num_features

    def test_num_test_points(self) -> None:
        """Test that the number of testing points is returned correctly."""
        assert self.dataset.num_testing_points == self.num_test_points

    def test_num_train_points(self) -> None:
        """Test that the number of testing points is returned correctly."""
        assert self.dataset.num_training_points == self.num_train_points

    def test_num_points(self) -> None:
        """Test that the total number of points is returned correctly."""
        assert self.dataset.num_points == self.num_test_points + self.num_train_points


class TestEmptyDataset:
    """Tests for the `EmptyDataset` class."""

    def test_members(self):
        """Test that all the data members of `EmptyDataset` are in fact empty."""
        dataset = EmptyDataset()
        assert not dataset.train_x
        assert not dataset.train_x_std
        assert not dataset.train_y
        assert not dataset.train_y_std
        assert not dataset.test_x
        assert not dataset.test_x_std
        assert not dataset.test_y
        assert not dataset.test_y_std

    def test_properties(self):
        """Test that the number-of-points properties correctly report that the dataset has zero points."""
        dataset = EmptyDataset()
        assert dataset.num_testing_points == 0
        assert dataset.num_training_points == 0
        assert dataset.num_points == 0

    def test_num_features(self):
        """Test that we can set the number of features, while still not having any points in the dataset."""
        dataset = EmptyDataset(num_features=5)
        assert dataset.num_features == 5
        assert dataset.num_points == 0
