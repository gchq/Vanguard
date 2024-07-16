"""
Tests for the vanguard.utils module.
"""

import unittest

import numpy as np
import numpy.typing

from tests.cases import get_default_rng
from vanguard.utils import UnseededRandomWarning, add_time_dimension, optional_random_generator


class OptionalRandomGeneratorTests(unittest.TestCase):
    """Tests for the `optional_random_generator` function."""

    def test_passing_none_warns(self) -> None:
        """Test that passing None warns the user and provides a new random generator."""
        with self.assertWarns(UnseededRandomWarning):
            rng = optional_random_generator(None)

        self.assertIsInstance(rng, np.random.Generator)

    def test_passing_generator_returns_generator(self) -> None:
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
        """Test functionality when there is no batch dimension in the data."""
        data = self.rng.standard_normal((self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertEqual(augmented_data.shape, (self.n_timesteps, self.n_dims + 1))

    def test_no_batch_monotonic(self) -> None:
        """Test the added time dimension is monotonic."""
        data = self.rng.standard_normal((self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertMonotonic(augmented_data[:, 0])

    def test_1_batch_shape(self) -> None:
        """Test the shape of the data output from `add_time_dimension` is as expected when given a batch dimension."""
        data = self.rng.standard_normal((self.batch_dim[0], self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertEqual(augmented_data.shape, (self.batch_dim[0], self.n_timesteps, self.n_dims + 1))

    def test_1_batch_monotonic(self) -> None:
        """Test the added time dimension is monotonic when there is a batch dimension in the data."""
        data = self.rng.standard_normal((self.batch_dim[0], self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertMonotonic(augmented_data[0, :, 0])

    def test_1_batch_equal(self) -> None:
        """Test the data with the time dimension added is as expected"""
        data = self.rng.standard_normal((self.batch_dim[0], self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertTrue((augmented_data[..., 0] == augmented_data[0, :, 0]).all())

    def test_2_batch_shape(self) -> None:
        """Test the shape of the data output from `add_time_dimension` is as expected with two batch dimensions."""
        data = self.rng.standard_normal((*self.batch_dim, self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertEqual(augmented_data.shape, self.batch_dim + (self.n_timesteps, self.n_dims + 1))

    def test_2_batch_monotonic(self) -> None:
        """Test the added time dimension is monotonic when there are two batch dimensions in the data."""
        data = self.rng.standard_normal((*self.batch_dim, self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertMonotonic(augmented_data[0, 0, :, 0])

    def test_2_batch_equal(self) -> None:
        """Test the data with the time dimension added is as expected when there are two batch dimensions."""
        data = self.rng.standard_normal((*self.batch_dim, self.n_timesteps, self.n_dims))
        augmented_data = add_time_dimension(data)
        self.assertArrayEqualAcrossDimensions(augmented_data[..., 0], augmented_data[0, 0, :, 0])

    def test_normalise_false(self) -> None:
        """Test the added time dimension is as expected when not applying normalisation."""
        data = self.rng.standard_normal((self.n_timesteps, self.n_dims))

        # Create augmented data both normalised and not
        augmented_data_raw = add_time_dimension(data, normalise=False)
        augmented_data_normalised = add_time_dimension(data, normalise=True)

        # The time data added should not match when the normalise flag is changed
        self.assertRaises(
            AssertionError, np.testing.assert_array_equal, augmented_data_raw[:, 0], augmented_data_normalised[:, 0]
        )

        # The added time data should still be monotonic
        self.assertMonotonic(augmented_data_raw[:, 0])
        self.assertMonotonic(augmented_data_normalised[:, 0])

        # In the case of normalised data, the steps should be equally spaced from 0 to 1
        np.testing.assert_array_equal(augmented_data_raw[:, 0], np.linspace(0, 1, self.n_timesteps))

    def assertArrayEqualAcrossDimensions(  # pylint: disable=invalid-name
        self,
        array_1: numpy.typing.NDArray,
        array_2: numpy.typing.NDArray,
    ) -> None:
        """
        Assert that two arrays are equal across all dimensions.

        :param array_1: First array to consider
        :param array_2: Second array to consider
        """
        self.assertTrue((array_1 == array_2).all())

    def assertMonotonic(self, array: numpy.typing.NDArray) -> None:  # pylint: disable=invalid-name
        """
        Assert that a given array is monotonic.

        :param array: Array we wish to check is monotonic
        """
        np.testing.assert_array_less(array[:-1], array[1:])
