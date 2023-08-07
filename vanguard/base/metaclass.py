"""
Contains the _StoreInitValues metaclass.
"""
import inspect


class _StoreInitValues(type):
    """
    A metaclass to store initialisation values.

    When this metaclass is applied to a class, the parameters passed to ``__init__``
    will be stored in the :py:attr:`_init_params` attribute.
    """
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        init_signature = inspect.signature(instance.__init__)
        init_params_as_kwargs = init_signature.bind_partial(*args, **kwargs).arguments
        instance._init_params = init_params_as_kwargs
        return instance
