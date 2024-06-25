"""
Contain some small utilities of use in some cases.
"""

from typing import Any, Generator, Tuple

import numpy as np
import numpy.typing
import torch

from .warnings import _RE_INCORRECT_LIKELIHOOD_PARAMETER


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
    batch_size: int, device: torch.DeviceObjType, *tensor_axis_pairs: Tuple[torch.Tensor, int]
) -> Generator[torch.Tensor, None, None]:
    """
    Return a never-ending generator that return random mini-batches of tensors with a shared first dimension.

    :param tensor_axis_pairs: Any number of (tensor, axis) pairs, where each tensor
        is of shape (n, ...), where n is shared between tensors, and ``axis`` denotes the axis along which
        the tensor should be batched. If an axis is out of range, the maximum axis value is used instead.
    :returns: A tensor generator.
    """
    first_tensor, first_axis = tensor_axis_pairs[0]
    first_tensor_length = first_tensor.shape[first_axis]

    if batch_size is None:
        batch_size = first_tensor_length

        def shuffle(array: numpy.typing.NDArray) -> None:  # pylint: disable=unused-argument
            """Identity shuffle function."""
    else:

        def shuffle(array: numpy.typing.NDArray) -> None:
            """Random shuffle function."""
            np.random.shuffle(array)

    index = 0
    indices = np.arange(first_tensor_length)
    shuffle(indices)
    while True:
        batch_indices = indices[index : index + batch_size]
        batch_tensors = []
        for tensor, axis in tensor_axis_pairs:
            multi_axis_slice = [slice(None, None, None) for _ in tensor.shape]
            multi_axis_slice[min(axis, len(tensor.shape) - 1)] = batch_indices
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
