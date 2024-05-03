"""
Contains the LearnYNoise decorator.
"""
from __future__ import annotations

import re
import warnings
from typing import Tuple, Type, TypeVar, Union

import numpy as np
import numpy.typing
import torch

from .base import GPController
from .decoratorutils import Decorator, process_args, wraps_class

ControllerT = TypeVar("ControllerT", bound=GPController)

_RE_NOT_LEARN_ERROR = re.compile(r"__init__\(\) got an unexpected keyword argument 'learn_additional_noise'")


class LearnYNoise(Decorator):
    """
    Learn the likelihood noise.

    This decorator passes the appropriate arguments to allow a :class:`~vanguard.base.gpcontroller.GPController`
    class to set the likelihood noise as unknown and subsequently learn it.

    :Example:
        >>> @LearnYNoise()
        ... class NewController(GPController):
        ...     pass
    """
    def __init__(self, **kwargs):
        """
        Initialise self.

        :param kwargs: Keyword arguments passed to :class:`~vanguard.decoratorutils.basedecorator.Decorator`.
        """
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        decorator = self

        @wraps_class(cls)
        class InnerClass(cls):
            """
            A wrapper for unknown, and hence learned, likelihood noise.
            """
            def __init__(self, *args, **kwargs):

                try:
                    all_parameters_as_kwargs = process_args(super().__init__, *args, y_std=0, **kwargs)
                except TypeError:
                    all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)

                all_parameters_as_kwargs.pop("self")
                y = all_parameters_as_kwargs["train_y"]
                y_std = _process_y_std(all_parameters_as_kwargs.pop("y_std", 0), y.shape, super().dtype, super().device)

                try:
                    train_x = all_parameters_as_kwargs.pop("train_x")
                except KeyError as error:
                    raise RuntimeError from error

                likelihood_kwargs = all_parameters_as_kwargs.pop("likelihood_kwargs", {})
                likelihood_kwargs["learn_additional_noise"] = True

                try:
                    super().__init__(train_x=train_x, likelihood_kwargs=likelihood_kwargs,
                                     y_std=y_std, **all_parameters_as_kwargs)
                except TypeError as error:
                    cannot_learn_y_noise = bool(_RE_NOT_LEARN_ERROR.match(str(error)))
                    if cannot_learn_y_noise:
                        likelihood_class = all_parameters_as_kwargs["likelihood_class"]
                        warnings.warn(f"Cannot learn additional noise for '{likelihood_class.__name__}'. "
                                      f"Consider removing the '{type(decorator).__name__}' decorator.")

                        likelihood_kwargs.pop("learn_additional_noise")
                        super().__init__(train_x=train_x, likelihood_kwargs=likelihood_kwargs,
                                         y_std=y_std, **all_parameters_as_kwargs)
                    else:
                        raise

        return InnerClass


def _process_y_std(y_std: Union[float, numpy.typing.NDArray[np.floating]], shape: Tuple[int], dtype: type, device: torch.DeviceObjType) -> torch.Tensor:
    """
    Create default y_std value or make sure given value is a tensor of the right type and shape.

    :param y_std: Values to use for standard deviations of data.
    :param shape: Shape of the output tensor to produce.
    :param dtype: Datatype of the output tensor produced.
    :param device: Torch device to place tensor on.
    :return: Tensor with each element being the standard deviation values defined in y_std.
    """
    tensor_value = torch.as_tensor(y_std, dtype=dtype, device=device)
    if tensor_value.shape == torch.Size([]):
        tensor_value = tensor_value * torch.ones(shape, dtype=dtype, device=device).squeeze(dim=-1)
    return tensor_value
