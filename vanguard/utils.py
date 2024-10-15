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
Contain some small utilities of use in some cases.
"""

import functools
import os
import warnings
from typing import Any, Callable, Generator, List, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing
import torch

from vanguard.warnings import _RE_INCORRECT_LIKELIHOOD_PARAMETER

if torch.cuda.is_available():  # pragma: no cover
    default_device = torch.device("cuda")
else:
    default_device = torch.device("cpu")


class DummyDistribution:
    """
    Empty mixin class for dummy distributions, used for isinstance() checks.

    Prefer checking against this rather than directly checking against e.g.
    :class:`~vanguard.classification.models.DummyKernelDistribution`, to avoid circular import issues.
    """


def add_time_dimension(data: np.typing.NDArray, normalise: bool = True) -> np.typing.NDArray:
    """
    Add an equal sample spacing dummy time dimension to some time series data.

    Required for signature kernel if no path parametrisation dimension is provided.
    The time dimension can also be normalised so that the sum of its
    squares is unity. The normalisation is irrelevant mathematically,
    but this choice leads to greater numerical stability for long
    time series.

    :param data: The time series of shape (..., n_timesteps, n_dimensions).
    :param normalise: Whether to normalise time as above.

    :returns: data but with new time dimension as the first dimension
                (..., n_timesteps, n_dimension + 1)
    """
    time_steps = data.shape[-2]
    if normalise:
        time_normalisation = np.sqrt(time_steps * (time_steps - 1) * (2 * time_steps - 1) / 6)
        final_value = (time_steps - 1) / time_normalisation
    else:
        final_value = 1
    time_variable = np.linspace(0, final_value, time_steps)
    tiled_time_variable = np.tile(time_variable, data.shape[:-2] + (1,))
    stackable_time_variable = np.expand_dims(tiled_time_variable, axis=-1)
    return np.concatenate([stackable_time_variable, data], axis=-1)


def instantiate_with_subset_of_kwargs(cls, **kwargs):
    """
    Instantiate a class with a kwargs, where some may not be required.

    This is useful if you intend to vary a class which may not need all
    of the parameters you wish to pass.

    :param cls: The class to be instantiated.
    :param kwargs: A set of keyword arguments containing a subset of arguments
                   which will successfully instantiate the class.

    :Example:
        >>> class MyClass:
        ...
        ...     def __init__(self, a, b):
        ...         self.a, self.b = a, b
        >>>
        >>> MyClass(a=1, b=2, c=3)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TypeError: __init__() got an unexpected keyword argument 'c'
        >>> x = instantiate_with_subset_of_kwargs(MyClass, a=1, b=2, c=3)
        >>> x.a, x.b
        (1, 2)

    When a parameter is missing (i.e. there is no valid subset of the passed kwargs),
    then the function behaves as expected:

    :Example:
        >>> class MyClass:
        ...
        ...     def __init__(self, a, b):
        ...         self.a, self.b = a, b
        >>>
        >>> instantiate_with_subset_of_kwargs(MyClass, a=1, c=3) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TypeError: __init__() missing 1 required positional argument: 'b'
    """
    remaining_kwargs = kwargs.copy()
    while remaining_kwargs:
        try:
            instance = cls(**remaining_kwargs)
        except TypeError as type_error:
            incorrect_likelihood_parameter_passed = _RE_INCORRECT_LIKELIHOOD_PARAMETER.match(str(type_error))
            try:
                incorrect_parameter = incorrect_likelihood_parameter_passed.group(1)
            except AttributeError as exc:
                raise type_error from exc
            else:
                remaining_kwargs.pop(incorrect_parameter)
        else:
            return instance
    return cls()


def infinite_tensor_generator(
    batch_size: Optional[int],
    device: torch.device,
    *tensor_axis_pairs: Tuple[torch.Tensor, int],
    rng: Optional[np.random.Generator] = None,
) -> Generator[torch.Tensor, None, None]:
    """
    Return a never-ending generator that return random mini-batches of tensors with a shared first dimension.

    :param tensor_axis_pairs: Any number of (tensor, axis) pairs, where each tensor
        is of shape (n, ...), where n is shared between tensors, and ``axis`` denotes the axis along which
        the tensor should be batched. If an axis is out of range, the maximum axis value is used instead.
    :param rng: Generator instance used to generate random numbers.
    :returns: A tensor generator.
    """
    rng = optional_random_generator(rng)

    # Validation - can't handle 0-dimensional tensors
    for tensor, axis in tensor_axis_pairs:
        if tensor.ndim == 0:
            raise ValueError(f"0-dimensional tensors are incompatible with infinite_tensor_generator. Got {tensor}")

    first_tensor, first_axis = tensor_axis_pairs[0]
    first_tensor_length = first_tensor.shape[first_axis]

    if batch_size is None:
        batch_size = first_tensor_length

        def shuffle(array: numpy.typing.NDArray) -> None:  # pylint: disable=unused-argument
            """Identity shuffle function."""
    else:

        def shuffle(array: numpy.typing.NDArray) -> None:
            """Random shuffle function."""
            rng.shuffle(array)

    index = 0
    indices = np.arange(first_tensor_length)
    shuffle(indices)
    while True:
        batch_indices = indices[index : index + batch_size]
        batch_tensors = []
        for tensor, axis in tensor_axis_pairs:
            multi_axis_slice = [slice(None, None, None) for _ in tensor.shape]
            multi_axis_slice[min(axis, max(0, tensor.ndim - 1))] = batch_indices
            batch_tensor = tensor[tuple(multi_axis_slice)]
            batch_tensors.append(batch_tensor.to(device))

        batch_tensors = tuple(batch_tensors)
        index += batch_size
        if index >= len(indices):
            rollovers = indices[index - batch_size :]
            indices = indices[: index - batch_size]
            shuffle(indices)
            indices = np.concatenate([rollovers, indices])
            index = 0
        yield batch_tensors


def generator_append_constant(generator: Generator[tuple, None, None], constant: Any) -> Generator[tuple, None, None]:
    """
    Augment a generator of tuples by appending a fixed item to each tuple.

    :param generator: The generator to augment.
    :param constant: The fixed element to append to each tuple in the generator.
    """
    for item in generator:
        yield item + (constant,)


class UnseededRandomWarning(UserWarning):
    """Warning for when unseeded random generators are used."""


def optional_random_generator(generator: Optional[np.random.Generator]) -> np.random.Generator:
    """
    Return the generator as-is, or a default unseeded one if :data:`None` is given.

    Warns if a default unseeded generator is used in testing.

    :param generator: If not None, returned as-is. If this _is_ None, and the code is running in a Pytest session,
        raise a warning reminding the user to seed their RNGs.
    :return: Either the given RNG (if not None) or a default unseeded RNG.
    """
    if generator is not None:
        return generator

    # Note that __debug__ must be first for this branch to be compiled out with `-O` (and it's only compiled out on
    # Python 3.10+) - but the performance penalty from not compiling it out should be negligible.
    if __debug__ and os.environ.get("PYTEST_VERSION") is not None:
        warnings.warn(
            "Using default unseeded RNG. Please seed your generators for consistent results!",
            stacklevel=4,
            category=UnseededRandomWarning,
        )

    return np.random.default_rng()


T = TypeVar("T")


def compose(functions: List[Callable[[T], T]]) -> Callable[[T], T]:
    """
    Compose a list of functions.

    Given a list of functions of type T -> T, returns a single function of type T -> T.
    For example, compose([a, b, c])(x) = a(b(c(x))).

    :Example:
        >>> def a(x):
        ...     return f"a({x})"
        >>> def b(x):
        ...     return f"b({x})"
        >>> def c(x):
        ...     return f"c({x})"
        >>> composed = compose([a, b, c])
        >>> composed("x")
        'a(b(c(x)))'

    :param functions: A list of functions of type (T -> T) to compose.
    :return: A single function of type (T -> T) that applies each of the passed functions in series.
    """
    return lambda x: functools.reduce(lambda acc, f: f(acc), reversed(functions), x)
