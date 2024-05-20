"""
The :class:`DisableStandardScaling` decorator will disable the default input standard scaling.
"""
from typing import Type, TypeVar

from .base import GPController
from .decoratorutils import Decorator, wraps_class

ControllerT = TypeVar("ControllerT", bound=GPController)


class DisableStandardScaling(Decorator):
    """
    Disable the default input scaling.

    :Example:
        >>> import numpy as np
        >>> from vanguard.kernels import ScaledRBFKernel
        >>> from vanguard.vanilla import GaussianGPController
        >>> from vanguard.standardise import DisableStandardScaling
        >>>
        >>> @DisableStandardScaling()
        ... class NoScaleController(GaussianGPController):
        ...     pass
        >>>
        >>> controller = NoScaleController(
        ...                     train_x=np.array([0, 1, 2, 3]),
        ...                     train_x_std=1,
        ...                     train_y=np.array([0, 1, 4, 9]),
        ...                     y_std=0.5,
        ...                     kernel_class=ScaledRBFKernel
        ...                     )
    """

    def __init__(self, **kwargs):
        """
        Initialise self.

        :param kwargs: Keyword arguments passed to :class:`~vanguard.decoratorutils.basedecorator.Decorator`.
        """
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        @wraps_class(cls)
        class InnerClass(cls):
            """
            A wrapper for disabling standard scaling.
            """

            def _input_standardise_modules(self, *modules):
                return modules

        return InnerClass
