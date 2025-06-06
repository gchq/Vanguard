# © Crown Copyright GCHQ
#
# Licensed under the GNU General Public License, version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for the vanguard.utils module.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.typing
import pytest
import torch
from typing_extensions import ContextManager

from tests.cases import get_default_rng
from vanguard.utils import (
    UnseededRandomWarning,
    add_time_dimension,
    generator_append_constant,
    infinite_tensor_generator,
    instantiate_with_subset_of_kwargs,
    multi_context,
    optional_random_generator,
)


class ExampleClass:
    """
    Example class to use with testing validation of input keyword arguments.
    """

    def __init__(self, a: str, b: int, c: float):
        """
        Initialise self.
        """
        self.a = a
        if isinstance(b, float):
            raise TypeError("__init__() got an unexpected keyword argument 'b'")
        self.b = b
        self.c = c


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


class TestClassCreation(unittest.TestCase):
    """
    Test creation of classes with keyword arguments.
    """

    def test_instantiate_with_subset_of_kwargs_all_valid(self) -> None:
        """Test `instantiate_with_subset_of_kwargs` behaviour with valid inputs."""
        a_val = "xyz"
        b_val = 1
        c_val = 1.0

        # Create the object
        output = instantiate_with_subset_of_kwargs(ExampleClass, a=a_val, b=b_val, c=c_val)

        # Verify the object is as expected
        self.assertEqual(output.a, a_val)
        self.assertEqual(output.b, b_val)
        self.assertEqual(output.c, c_val)

    def test_instantiate_with_subset_of_kwargs_type_error(self) -> None:
        """Test `instantiate_with_subset_of_kwargs` when the class raises a `TypeError` upon creation."""
        a_val = "xyz"
        b_val = 1.0
        c_val = 1.0

        # Create the object - we expect a type error to be raised upon failed creation due to b_val not being
        # a float
        with self.assertRaises(TypeError):
            instantiate_with_subset_of_kwargs(ExampleClass, a=a_val, b=b_val, c=c_val)


class TestGenerators(unittest.TestCase):
    """
    Test generator usage.
    """

    def test_infinite_tensor_generator_with_batch_size(self) -> None:
        """Test `infinite_tensor_generator` when a batch size is provided."""
        batch_size = 2
        device = torch.device("cpu")
        rng = get_default_rng()
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Create an infinite tensor generator and sample the first tensor from it
        result = infinite_tensor_generator(
            batch_size,
            device,
            (tensor, 0),
            rng=rng,
        )

        # Verify the output - we have batch_size elements per sample, until we reach the entire tensor provided.
        # In our case, we expect the first sample to have batch_size (2) rows and the second sample to have the
        # remaining (1) row from the provided tensor.
        sample_1 = next(result)
        sample_2 = next(result)
        self.assertEqual(len(sample_1), 1)
        self.assertListEqual(list(sample_1[0].shape), [batch_size, tensor.shape[1]])
        self.assertListEqual(list(sample_2[0].shape), [tensor.shape[0] - batch_size, tensor.shape[1]])

    def test_infinite_tensor_generator_without_batch_size(self) -> None:
        """Test `infinite_tensor_generator` when a batch size is not provided."""
        batch_size = None
        device = torch.device("cpu")
        rng = get_default_rng()
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Create an infinite tensor generator and sample the first tensor from it
        result = infinite_tensor_generator(
            batch_size,
            device,
            (tensor, 0),
            rng=rng,
        )

        # The generator output should be the same size as the input since we are not using batches
        sample_1 = next(result)
        self.assertEqual(len(sample_1), 1)
        self.assertListEqual(list(sample_1[0].shape), list(tensor.shape))

    def test_generator_append_constant(self) -> None:
        """Test appending a constant to a generator output."""
        generator = (i * np.ones([3, 1]) for i in range(3))
        constant = 22

        # Define expected output
        expected_output = [
            np.array([22.0, 22.0, 22.0]).reshape(-1, 1),
            np.array([23.0, 23.0, 23.0]).reshape(-1, 1),
            np.array([24.0, 24.0, 24.0]).reshape(-1, 1),
        ]

        # Call the function, which we expect to append the given constant to each generator output
        output = [i for i in generator_append_constant(generator=generator, constant=constant)]

        # Verify outputs match
        self.assertEqual(len(output), 3)
        for index in range(3):
            np.testing.assert_array_equal(output[index], expected_output[index])

    def test_generator_zero_dimensional(self):
        """Test that an appropriate error is raised if 0-dimensional tensors are passed to infinite_tensor_generator."""
        batch_size = 2
        device = torch.device("cpu")
        rng = get_default_rng()
        tensor = torch.tensor(1)  # 0-dimensional

        # Create an infinite tensor generator and sample the first tensor from it
        generator = infinite_tensor_generator(
            batch_size,
            device,
            (tensor, 0),
            rng=rng,
        )

        with pytest.raises(ValueError, match="0-dimensional tensors are incompatible"):
            next(generator)


@pytest.mark.parametrize("num_contexts", [1, 5])
def test_multi_context(num_contexts: int):
    """
    Test the `multi_context` context manager.

    Given that the context manager syntax is ultimately just syntactic sugar for calling `ctx.__enter__` and
    `ctx.__exit__` (with some extra semantics around exception handling), all we really need to do is check that
    these methods are called on each context manager passed to `multi_context` at the appropriate times. We don't
    test the specifics of exception propagation, as that is (a) too complex for a single unit test, and (b) more the
    responsibility of the `contextlib.ExitStack` that `multi_context` uses internally.

    Test with both a single context and multiple contexts being passed to `multi_context`.
    """
    dummy_contexts = [MagicMock(spec=ContextManager)() for _ in range(num_contexts)]
    context = multi_context(dummy_contexts)

    for ctx in dummy_contexts:
        # check that we haven't entered or left any of the contexts yet
        ctx.__enter__.assert_not_called()
        ctx.__exit__.assert_not_called()

    # enter the multi context
    with context:
        for ctx in dummy_contexts:
            # check that we have entered but not left each of the contexts yet
            ctx.__enter__.assert_called_once()
            ctx.__exit__.assert_not_called()

    for ctx in dummy_contexts:
        # check that we have left each of the contexts
        ctx.__enter__.assert_called_once()
        ctx.__exit__.assert_called_once()
