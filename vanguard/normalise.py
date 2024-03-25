"""
The :class:`NormaliseY` decorator will scale the y-inputs to a unit normal distribution.
"""
import numpy as np
import torch

from .base import GPController
from .decoratorutils import Decorator, process_args, wraps_class
from typing import TypeVar, Type
from .base.posteriors import Posterior

ControllerT = TypeVar("ControllerT", bound=GPController)


class NormaliseY(Decorator):
    r"""
    Normalise the y-inputs and the variance.

    Let :math:`{\bf y}` and :math:`{\bf\psi}` denote ``train_y`` and ``y_std`` respectively.  If the values in
    :math:`{\bf y}` have empirical mean :math:`\overline{Y}` and variance  :math:`s^2`, then this decorator will
    compute the following transformations before initialisation:

    .. math::

        {\bf y} \mapsto \frac{{\bf y} - \overline{Y}}{s}, \quad {\bf\psi} \mapsto \ \frac{{\bf\psi}}{s}.

    The inverse transformation is then applied to the return values of posterior methods.

    :Example:
        >>> import numpy as np
        >>> from vanguard.kernels import ScaledRBFKernel
        >>> from vanguard.vanilla import GaussianGPController
        >>>
        >>> @NormaliseY(ignore_methods=("__init__",))
        ... class NormalisedController(GaussianGPController):
        ...     pass
        >>>
        >>> controller = NormalisedController(
        ...                     train_x=np.array([0, 1, 2, 3]),
        ...                     train_x_std=1,
        ...                     train_y=np.array([0, 1, 4, 9]),
        ...                     y_std=0.5,
        ...                     kernel_class=ScaledRBFKernel
        ...                     )
        >>> controller.train_y.T
        tensor([[-1.0000, -0.7143,  0.1429,  1.5714]])
    """
    def __init__(self, **kwargs):
        """
        Initialise self.

        :param kwargs: Keyword arguments passed to :class:`~vanguard.decoratorutils.basedecorator.Decorator`.
        """
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: Type[ControllerT]) -> ControllerT:
        @wraps_class(cls)
        class InnerClass(cls):
            """
            A wrapper for normalising y inputs and variance.
            """
            def __init__(self, *args, **kwargs):

                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                all_parameters_as_kwargs.pop("self")

                y_std = all_parameters_as_kwargs.pop("y_std")
                train_x = all_parameters_as_kwargs.pop("train_x")
                train_y = all_parameters_as_kwargs.pop("train_y")

                if y_std is None:
                    n, *_ = train_x.shape
                    y_std = torch.zeros(n, dtype=self.dtype, device=self.device)

                _normalising_mean, _normalising_std = train_y.mean(), train_y.std()
                train_y = (train_y - _normalising_mean) / _normalising_std
                y_std = y_std / _normalising_std

                def normalise_posterior_class(posterior_class: Type[Posterior]) -> Posterior:
                    """Wrap a posterior class to enable normalisation."""
                    @wraps_class(posterior_class)
                    class NormalisedPosterior(posterior_class):
                        """
                        Un-scale the distribution at initialisation.
                        """
                        def prediction(self) -> tuple[torch.Tensor, torch.Tensor]:
                            """
                            Un-normalise values.

                            :return: The mean and covariance, on the original scale of the data.
                            """
                            mean, covar = super().prediction()
                            unscaled_mean = mean * _normalising_std + _normalising_mean
                            unscaled_covar = covar * _normalising_std ** 2
                            return unscaled_mean, unscaled_covar

                        def confidence_interval(self, alpha: float = 0.05) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                            """
                            Un-normalise values.

                            :param alpha: Confidence threshold that interval should cover. The resulting confidence interval
                                should contain the underlying data with probability :math:`1 - \alpha`.
                            :return: The mean and covariance, on the original scale of the data.
                            """
                            mean, lower, upper = super().confidence_interval(alpha)
                            unscaled_mean = (mean * _normalising_std) + _normalising_mean
                            unscaled_lower = (lower * _normalising_std) + _normalising_mean
                            unscaled_upper = (upper * _normalising_std) + _normalising_mean
                            return unscaled_mean, unscaled_lower, unscaled_upper

                        def log_probability(self, y : torch.Tensor) -> torch.Tensor:
                            """
                            Apply the change of variables to the density using the normalise map.

                            :param y: The data-point at which to evaluate the log-probability
                            :return: The log-probability of some data-point ``y``
                            """
                            normalised_y = (y - _normalising_mean) / _normalising_std
                            norm_map_deriv_values = np.ones_like(y) / _normalising_std
                            jacobian = np.sum(np.log(np.abs(norm_map_deriv_values)))
                            return jacobian + super().log_probability(normalised_y)

                        def sample(self, n_samples: int = 1) -> torch.Tensor:
                            """
                            Generate samples from posterior.

                            :param n_samples: The number of samples to generate.
                            :return: Array of samples.
                            """
                            normalised_samples = super().sample(n_samples=n_samples)
                            return normalised_samples * _normalising_std + _normalising_mean

                    return NormalisedPosterior

                self.posterior_class = normalise_posterior_class(self.posterior_class)
                self.posterior_collection_class = normalise_posterior_class(self.posterior_collection_class)

                super().__init__(train_x=train_x, train_y=train_y, y_std=y_std, **all_parameters_as_kwargs)

            @staticmethod
            def warn_normalise_y():
                """Override base warning because y normalisation has been applied."""
                pass

        return InnerClass
