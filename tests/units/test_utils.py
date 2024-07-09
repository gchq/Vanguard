"""
Tests for the vanguard.utils module.
"""

import unittest

import numpy as np
import numpy.typing

from tests.cases import get_default_rng
from vanguard.utils import UnseededRandomWarning, add_time_dimension, optional_random_generator


class OptionalRandomGeneratorTests(unittest.TestCase):
    """Tests for the optional_random_generator() function."""

    def test_passing_none_warns(self):
        """Test that passing None warns the user and provides a new random generator."""
        with self.assertWarns(UnseededRandomWarning):
            rng = optional_random_generator(None)

        self.assertIsInstance(rng, np.random.Generator)

    def test_passing_generator_returns_generator(self):
        """Test that passing a random generator returns the same generator unaltered."""
        rng = get_default_rng()
        rng_from_optional = optional_random_generator(rng)
        self.assertIs(rng, rng_from_optional)


class TimeDimensionTests(unittest.TestCase):
    """
    Tests for the add_time_dimension function.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.n_timesteps = 11
        self.n_dims = 3
        self.batch_dim = (23, 29)
        self.rng = get_default_rng()

    def test_no_batch_shape(self) -> None:
        data = self.rng.standard_normal((self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertEqual(augmented_data.shape, (self.n_timesteps, self.n_dims + 1))

    def test_no_batch_monotonic(self) -> None:
        data = self.rng.standard_normal((self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertMonotonic(augmented_data[:, 0])

    def test_1_batch_shape(self) -> None:
        data = self.rng.standard_normal((self.batch_dim[0], self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertEqual(augmented_data.shape, (self.batch_dim[0], self.n_timesteps, self.n_dims + 1))

    def test_1_batch_monotonic(self) -> None:
        data = self.rng.standard_normal((self.batch_dim[0], self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertMonotonic(augmented_data[0, :, 0])

    def test_1_batch_equal(self) -> None:
        data = self.rng.standard_normal((self.batch_dim[0], self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertTrue((augmented_data[..., 0] == augmented_data[0, :, 0]).all())

    def test_2_batch_shape(self) -> None:
        data = self.rng.standard_normal((*self.batch_dim, self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertEqual(augmented_data.shape, self.batch_dim + (self.n_timesteps, self.n_dims + 1))

    def test_2_batch_monotonic(self) -> None:
        data = self.rng.standard_normal((*self.batch_dim, self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertMonotonic(augmented_data[0, 0, :, 0])

    def test_2_batch_equal(self) -> None:
        data = self.rng.standard_normal((*self.batch_dim, self.n_timesteps, self.n_dims))
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
