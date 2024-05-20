"""
Contains the SetWarp decorator.
"""

from typing import Any, Tuple, Type, TypeVar

import numpy as np
import numpy.typing
import torch

from ..base import GPController
from ..base.posteriors import Posterior
from ..decoratorutils import Decorator, process_args, wraps_class
from .basefunction import WarpFunction
from .intermediate import is_intermediate_warp_function

ControllerT = TypeVar("ControllerT", bound=GPController)


class SetWarp(Decorator):
    """
    Map a GP through a warp function.

    :Example:
            >>> from vanguard.base import GPController
            >>> from vanguard.warps.warpfunctions import BoxCoxWarpFunction
            >>>
            >>> @SetWarp(BoxCoxWarpFunction(1))
            ... class MyController(GPController):
            ...     pass
    """

    def __init__(self, warp_function: WarpFunction, **kwargs: Any):
        """
        Initialise self.

        :param warp_function: The warp function to be applied to the GP.
        :param kwargs: Keyword arguments passed to :class:`~vanguard.decoratorutils.basedecorator.Decorator`.
        """
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)
        self.warp_function = warp_function

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        warp_function = self.warp_function

        @wraps_class(cls)
        class InnerClass(cls):
            """
            A wrapper for applying a compositional warp to a controller class.
            """

            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, **kwargs)

                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                all_parameters_as_kwargs.pop("self")

                for warp_component in warp_function.components:
                    if is_intermediate_warp_function(warp_component):
                        warp_component.activate(**all_parameters_as_kwargs)

                warp_copy = warp_function.copy().float()
                self.warp = warp_copy
                self._smart_optimiser.register_module(self.warp)
                self.train_y = self.train_y.to(self.device)

                def _unwarp_values(
                    *values: numpy.typing.NDArray[np.floating],
                ) -> Tuple[numpy.typing.NDArray[np.floating], ...]:
                    """Map values back through the warp."""
                    values_as_tensors = (
                        torch.as_tensor(value, dtype=self.dtype, device=self.device) for value in values
                    )
                    unwarped_values_as_tensors = (warp_copy.inverse(tensor).squeeze() for tensor in values_as_tensors)
                    unwarped_values_as_arrays = tuple(
                        tensor.detach().cpu().numpy() for tensor in unwarped_values_as_tensors
                    )
                    return unwarped_values_as_arrays

                def _warp_values(
                    *values: numpy.typing.NDArray[np.floating],
                ) -> Tuple[numpy.typing.NDArray[np.floating], ...]:
                    """Map values through the warp."""
                    values_as_tensors = (
                        torch.as_tensor(value, dtype=self.dtype, device=self.device) for value in values
                    )
                    warped_values_as_tensors = (warp_copy(tensor).squeeze() for tensor in values_as_tensors)
                    warped_values_as_arrays = tuple(
                        tensor.detach().cpu().numpy() for tensor in warped_values_as_tensors
                    )
                    return warped_values_as_arrays

                def _warp_derivative_values(
                    *values: numpy.typing.NDArray[np.floating],
                ) -> Tuple[numpy.typing.NDArray[np.floating], ...]:
                    """Map values through the derivative of the warp."""
                    values_as_tensors = (
                        torch.as_tensor(value, dtype=self.dtype, device=self.device) for value in values
                    )
                    warped_values_as_tensors = (warp_copy.deriv(tensor).squeeze() for tensor in values_as_tensors)
                    warped_values_as_arrays = tuple(
                        tensor.detach().cpu().numpy() for tensor in warped_values_as_tensors
                    )
                    return warped_values_as_arrays

                def warp_posterior_class(posterior_class: Type[Posterior]) -> Type[Posterior]:
                    """Wrap a posterior class to enable warping."""

                    @wraps_class(posterior_class)
                    class WarpedPosterior(posterior_class):
                        """
                        Un-scale the distribution at initialisation.
                        """

                        def prediction(self) -> torch.tensor:  # pytest: ignore [reportGeneralTypeIssues]
                            """Un-warp values."""
                            raise TypeError("The mean and covariance of a warped GP cannot be computed exactly.")

                        def confidence_interval(
                            self, alpha: float = 0.05
                        ) -> Tuple[
                            numpy.typing.NDArray[np.floating],
                            numpy.typing.NDArray[np.floating],
                            numpy.typing.NDArray[np.floating],
                        ]:
                            """Un-warp values."""
                            mean, lower, upper = super().confidence_interval(alpha)
                            return _unwarp_values(mean, lower, upper)

                        def log_probability(
                            self, y: Tuple[numpy.typing.NDArray[np.floating]]
                        ) -> numpy.typing.NDArray[np.floating]:
                            """Apply the change of variables to the density using the warp."""
                            warped_y = _warp_values(y)
                            warp_deriv_values = _warp_derivative_values(y)
                            jacobian = np.sum(np.log(np.abs(warp_deriv_values)))
                            return jacobian + super().log_probability(warped_y)

                    return WarpedPosterior

                self.posterior_class = warp_posterior_class(self.posterior_class)
                self.posterior_collection_class = warp_posterior_class(self.posterior_collection_class)

            @classmethod
            def new(cls, instance: Type[ControllerT], **kwargs: Any) -> Type[ControllerT]:
                """Also apply warping to the new instance."""
                new_instance = super().new(instance, **kwargs)
                new_instance.warp = instance.warp
                new_instance._gp.train_targets = new_instance.warp(new_instance._gp.train_targets).squeeze(dim=-1)
                return new_instance

            def _sgd_round(self, n_iters: int = 100, gradient_every: int = 100) -> torch.Tensor:
                """Calculate loss and warp train_y."""
                loss = super()._sgd_round(n_iters=n_iters, gradient_every=gradient_every)
                warped_train_y = self.warp(self.train_y).squeeze(dim=-1)
                self._gp.train_targets = warped_train_y
                return loss

            def _unwarp_values(
                self, *values: numpy.typing.NDArray[np.floating]
            ) -> Tuple[numpy.typing.NDArray[np.floating], ...]:
                """Map values back through the warp."""
                values_as_tensors = (torch.as_tensor(value) for value in values)
                unwarped_values_as_tensors = (self.warp.inverse(tensor).reshape(-1) for tensor in values_as_tensors)
                unwarped_values_as_arrays = tuple(
                    tensor.detach().cpu().numpy() for tensor in unwarped_values_as_tensors
                )
                return unwarped_values_as_arrays

            def _loss(self, train_x: torch.Tensor, train_y: torch.Tensor) -> torch.Tensor:
                """Subtract additional derivative term from the mll."""
                warped_train_y = self.warp(train_y).squeeze(dim=-1)
                self._gp.train_targets = warped_train_y
                nmll = super()._loss(train_x, warped_train_y)
                return nmll - self.warp.deriv(train_y).squeeze(dim=-1).sum()

            @staticmethod
            def warn_normalise_y() -> None:
                """Override base warning because warping renders y normalisation unimportant."""
                pass

        return InnerClass
