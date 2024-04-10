"""
Contains decorators to deal with input features that aren't vectors.
"""
from functools import partial
from typing import Tuple, Type, TypeVar, Union

import numpy as np
import torch
from gpytorch import kernels

from .base import GPController
from .decoratorutils import Decorator, process_args, wraps_class

ControllerT = TypeVar("ControllerT", bound=GPController)


class HigherRankFeatures(Decorator):
    """
    Make a :class:`~vanguard.base.gpcontroller.GPController` compatible with higher rank features.

    GPyTorch assumes that input features are rank-1 (vectors) and a variety of
    RuntimeErrors are thrown from different places in the code if this is not true.
    This decorator modifies the gp model class to make it compatible with higher
    rank features.

    :Example:
        >>> @HigherRankFeatures(2)
        ... class NewController(GPController):
        ...     pass
    """
    def __init__(self, rank: int, **kwargs):
        """
        :param rank: The rank of the input features. Should be a positive integer.
        """
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)
        self.rank = rank

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        rank = self.rank

        @wraps_class(cls)
        class InnerClass(cls):
            def __init__(self, *args, **kwargs):
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                train_x = all_parameters_as_kwargs["train_x"]
                all_parameters_as_kwargs.pop("self")
                self.gp_model_class = _HigherRankFeaturesModel(train_x.shape[-rank:])(self.gp_model_class)
                kernel_class = all_parameters_as_kwargs.pop("kernel_class")
                new_kernel_class = _HigherRankFeaturesKernel(train_x.shape[-rank:])(kernel_class)

                super().__init__(kernel_class=new_kernel_class, **all_parameters_as_kwargs)
        return InnerClass


class _HigherRankFeaturesModel:
    """
    A decorator for a model, enabling higher rank features.

    GPyTorch assumes that input features are rank-1 (vectors) and a variety of
    RuntimeErrors are thrown from different places in the code if this is not true.
    This decorator can be applied to a GPyTorch model and deals with the feature
    shapes to avoid these issues. The decorator intercepts the training data
    and any data passed to ``__call__``, flattening it so that the shapes work out
    correctly. The data are then returned to their native shape before any actual
    computation (e.g. inside kernels) is performed.
    """
    def __init__(self, shape: Union[Tuple[int], torch.Size]):
        """
        :param shape: The native shape of a single data point.
        """
        self.shape = tuple(shape)
        self.flat_shape = np.prod(self.shape)

    def __call__(self, model_cls: Type[ControllerT]) -> ControllerT:
        shape = self.shape
        flat_shape = self.flat_shape
        _flatten = partial(self._flatten, item_shape=shape,
                           item_flat_shape=flat_shape)
        _unflatten = partial(self._unflatten, item_shape=shape)

        @wraps_class(model_cls)
        class InnerClass(model_cls):
            def __init__(self, train_x: torch.Tensor, *args, **kwargs):
                super().__init__(_flatten(train_x), *args, **kwargs)

            def __call__(self, *args, **kwargs):
                args = [_flatten(arg) for arg in args]
                return super().__call__(*args, **kwargs)

            def forward(self, x):
                return super().forward(_unflatten(x))

        return InnerClass

    @staticmethod
    def _flatten(tensor: torch.Tensor, item_shape: Tuple[int], item_flat_shape: int) -> torch.Tensor:
        """
        Reshapes tensors to flat (rank - 1) features.

        :param tensor: The tensor to reshape.
        :param item_shape: The native shape of a single item.
        :param item_flat_shape: The flatten length of a single item.

        :returns: Reshape tensor.
        """
        new_shape = tuple(tensor.shape[:-len(item_shape)])
        new_shape = new_shape + (item_flat_shape,)
        return tensor.reshape(new_shape)

    @staticmethod
    def _unflatten(tensor: torch.Tensor, item_shape: Tuple[int]) -> torch.Tensor:
        """
        Reshapes flatten tensors to native feature shape.

        :param tensor: The tensor to reshape.
        :param item_shape: The native shape of a single item.

        :returns: Reshape tensor.
        """
        new_shape = tuple(tensor.shape[:-1])
        new_shape = new_shape + item_shape
        return tensor.reshape(new_shape)


class _HigherRankFeaturesKernel(_HigherRankFeaturesModel):
    """
    A decorator for a kernel, enabling higher rank features.

    GPyTorch assumes that input features are rank-1 (vectors) and a variety of
    RuntimeErrors are thrown from different places in the code if this is not true.
    This decorator can be applied to a GPyTorch kernel and deals with the feature
    shapes to avoid these issues. In particular, kernels only pose an issue when
    using variational orthogonal features, in which the VOF basis has a forward
    method. Unlike the kernel forward method itself, which is safely behind calls
    to the model forward method, this method is exposed directly to the flattened
    data.
    """
    def __init__(self, shape: Union[Tuple[int], torch.Size]):
        """
        :param shape: The native shape of a single data point.
        """
        self.shape = tuple(shape)
        self.flat_shape = np.prod(self.shape)

    def __call__(self, kernel_cls: Type[kernels.Kernel]) -> Type[kernels.Kernel]:
        shape = self.shape
        _unflatten = partial(self._unflatten, item_shape=shape)

        @wraps_class(kernel_cls)
        class InnerClass(kernel_cls):
            def get_vof_basis(self, *args, **kwargs):
                basis_type = type(super().get_vof_basis(*args, **kwargs))

                @wraps_class(basis_type)
                class InnerBasisType(basis_type):
                    def forward(self, x, *args, **kwargs):
                        return super().forward(_unflatten(x), *args, **kwargs)
                return InnerBasisType(*args, **kwargs)

        return InnerClass
