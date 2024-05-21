"""
Tests for the vanguard.utils module.
"""

import unittest

import numpy as np
import numpy.typing

from vanguard.utils import add_time_dimension


class TimeDimensionTests(unittest.TestCase):
    """
    Tests for the add_time_dimension function.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.n_timesteps = 11
        self.n_dims = 3
        self.batch_dim = (23, 29)

    def test_no_batch_shape(self) -> None:
        data = np.random.randn(self.n_timesteps, self.n_dims)
        augmented_data = add_time_dimension(data)
        self.assertEqual(augmented_data.shape, (self.n_timesteps, self.n_dims + 1))

    def test_no_batch_monotonic(self) -> None:
        data = np.random.randn(self.n_timesteps, self.n_dims)
        augmented_data = add_time_dimension(data)
        self.assertMonotonic(augmented_data[:, 0])

    def test_1_batch_shape(self) -> None:
        data = np.random.randn(self.batch_dim[0], self.n_timesteps, self.n_dims)
        augmented_data = add_time_dimension(data)
        self.assertEqual(augmented_data.shape, (self.batch_dim[0], self.n_timesteps, self.n_dims + 1))

    def test_1_batch_monotonic(self) -> None:
        data = np.random.randn(self.batch_dim[0], self.n_timesteps, self.n_dims)
        augmented_data = add_time_dimension(data)
        self.assertMonotonic(augmented_data[0, :, 0])

    def test_1_batch_equal(self) -> None:
        data = np.random.randn(self.batch_dim[0], self.n_timesteps, self.n_dims)
        augmented_data = add_time_dimension(data)
        self.assertTrue((augmented_data[..., 0] == augmented_data[0, :, 0]).all())

    def test_2_batch_shape(self) -> None:
        data = np.random.randn(*self.batch_dim, self.n_timesteps, self.n_dims)
        augmented_data = add_time_dimension(data)
        self.assertEqual(augmented_data.shape, self.batch_dim + (self.n_timesteps, self.n_dims + 1))

    def test_2_batch_monotonic(self) -> None:
        data = np.random.randn(*self.batch_dim, self.n_timesteps, self.n_dims)
        augmented_data = add_time_dimension(data)
        self.assertMonotonic(augmented_data[0, 0, :, 0])

    def test_2_batch_equal(self) -> None:
        data = np.random.randn(*self.batch_dim, self.n_timesteps, self.n_dims)
        augmented_data = add_time_dimension(data)
        self.assertArrayEqualAcrossDimensions(augmented_data[..., 0], augmented_data[0, 0, :, 0])

    def assertArrayEqualAcrossDimensions(  # pylint: disable=invalid-name
        self,
        array_1: numpy.typing.NDArray,
        array_2: numpy.typing.NDArray,
    ) -> None:
        self.assertTrue((array_1 == array_2).all())

    def assertMonotonic(self, array: numpy.typing.NDArray) -> None:  # pylint: disable=invalid-name
        np.testing.assert_array_less(array[:-1], array[1:])
