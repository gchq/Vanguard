"""
Enable lazy initialisation in controllers.

Some warp functions require the input data passed to the controller class in
order to initialise properly. In order to avoid needing to set this ahead of time,
the :func:`require_controller_input` decorator will allow a warp function to be
initialised lazily, only becoming a full warp function upon activation.
"""
from typing import Any, Callable, Type, TypeVar

from ..decoratorutils import process_args, wraps_class
from .basefunction import WarpFunction

WarpFunctionT = TypeVar("WarpFunctionT", bound=WarpFunction)


def is_intermediate_warp_function(func: WarpFunction) -> bool:
    """
    Establish if a warp function is intermediate.

    :param func: A warp function instance which may be intermediate.
    :return: True, if the warp function is intermediate.
    """
    return hasattr(func, "CACHED_PARAMS_AS_KWARGS") and hasattr(func, "activate") and isinstance(func, WarpFunction)


def require_controller_input(cache_name: str) -> Callable[[Type[WarpFunctionT]], Type[WarpFunctionT]]:
    """
    Force a warp function to wrap lazily, so that it may take controller class input.

    :param cache_name: The name of the class attribute which will hold the input parameters.

    :Example:
        >>> import torch
        >>> from vanguard.warps.warpfunctions import AffineWarpFunction
        >>>
        >>> @require_controller_input("controller_inputs")
        ... class GaussianScaledAffineWarpFunction(AffineWarpFunction):
        ...     '''
        ...     Scale inputs by the mean and standard deviation
        ...     of the training data.
        ...     '''
        ...     def __init__(self):
        ...         train_y = self.controller_inputs["train_y"]
        ...         mu, sigma = train_y.mean(), train_y.std()
        ...         super().__init__(1/sigma, -mu/sigma)
        >>>
        >>> warp_function = GaussianScaledAffineWarpFunction()
        >>> warp_function(torch.Tensor([1]))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        AttributeError: ...
        >>> warp_function.activate(train_y=torch.Tensor([0, 1, 2, 3, 4]))
        >>> warp_function(torch.Tensor([1])).detach()
        tensor([[-0.6325]])

    .. note::
        The :class:`~vanguard.warps.SetWarp` decorator will call :meth:`activate` on the user's behalf,
        so in the majority of cases one should not worry about this step. Only key word arguments can be passed to
        :meth:`activate`.

    .. warning::
        Despite best efforts, failing to activate an intermediate warp function before usage can return opaque error
        messages, and so checking for failed activation should be a priority when debugging any sort of error
        surrounding usage.
    """

    def decorator(cls: Type[WarpFunctionT]) -> Type[WarpFunctionT]:
        """Return the intermediate warp function class."""

        @wraps_class(cls)
        class IntermediateClass(cls):
            """
            Lazily holds controller class input until activation.
            """

            CACHED_PARAMS_AS_KWARGS = {}
            setattr(cls, cache_name, {})
            __init__ = _only_cache_init_values(cls.__init__)

            def activate(self, **controller_input_as_kwargs: Any):
                """Activate the intermediate warp function."""
                controller_inputs_as_kwargs = getattr(self, cache_name)
                controller_inputs_as_kwargs.update(controller_input_as_kwargs)
                try:
                    super().__init__(**self.CACHED_PARAMS_AS_KWARGS)
                except KeyError as error:
                    error_message = "Activation failed: Make sure that you have passed all required parameters."
                    raise ValueError(error_message) from error

        return IntermediateClass

    return decorator


def _only_cache_init_values(init_method: Callable) -> Callable:
    """Wrap an __init__ method to only cache the input parameters."""

    def inner_func(self: WarpFunction, *args: Any, **kwargs: Any) -> None:
        """Cache the input parameters."""
        all_parameters_as_kwargs = process_args(init_method, self, *args, **kwargs)
        all_parameters_as_kwargs.pop("self")
        self.CACHED_PARAMS_AS_KWARGS.update(all_parameters_as_kwargs)

    return inner_func
