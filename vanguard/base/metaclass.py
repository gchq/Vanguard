"""
Contains the _StoreInitValues metaclass.
"""
import inspect
from typing import Any, Type, TypeVar

T = TypeVar('T')

class _StoreInitValues(type):
    """
    A metaclass to store initialisation values.

    When this metaclass is applied to a class, the parameters passed to ``__init__``
    will be stored in the :attr:`_init_params` attribute.
    """

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> Type[T]:
        instance = super().__call__(*args, **kwargs)
        init_signature = inspect.signature(instance.__init__)
        init_params_as_kwargs = init_signature.bind_partial(*args, **kwargs).arguments
        instance._init_params = init_params_as_kwargs
        return instance
