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
The :class:`NormaliseY` decorator will scale the y-inputs to a unit normal distribution.
"""

import warnings
from typing import Any, TypeVar

import torch
from typing_extensions import override

from vanguard import utils
from vanguard.base import GPController
from vanguard.base.gpcontroller import BaseGPController
from vanguard.base.posteriors import Posterior
from vanguard.classification.mixin import Classification, ClassificationMixin
from vanguard.decoratorutils import Decorator, process_args, wraps_class
from vanguard.decoratorutils.errors import BadCombinationWarning
from vanguard.variational import VariationalInference

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

    .. note::
        The empirical variance is the _sample_ empirical variance, not the population empirical variance (that is, it
        is :math:`\frac{1}{n-1}\sum_{y \in \bf y} (y - \overline{Y})^2`, where :math:`n` is the number of data points
        in ``train_y``).

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
        ...                     train_x=np.array([0.0, 1.0, 2.0, 3.0]),
        ...                     train_x_std=1.0,
        ...                     train_y=np.array([0.0, 1.0, 4.0, 9.0]),
        ...                     y_std=0.5,
        ...                     kernel_class=ScaledRBFKernel
        ...                     )
        >>> controller.train_y.T.cpu()
        tensor([[-0.8660, -0.6186,  0.1237,  1.3609]])
        >>> controller.train_y.mean().item()
        0.0
        >>> controller.train_y.std().item()
        1.0
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialise self.

        :param kwargs: Keyword arguments passed to :class:`~vanguard.decoratorutils.basedecorator.Decorator`.
        """
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    @property
    @override
    def safe_updates(self) -> dict[type, set[str]]:
        return self._add_to_safe_updates(
            super().safe_updates,
            {
                ClassificationMixin: {"classify_points", "classify_fuzzy_points"},
                VariationalInference: {"__init__", "_predictive_likelihood", "_fuzzy_predictive_likelihood"},
                Classification: {
                    "posterior_over_point",
                    "posterior_over_fuzzy_point",
                    "fuzzy_predictive_likelihood",
                    "predictive_likelihood",
                },
                BaseGPController: {
                    "_input_standardise_modules",
                    "_gp_forward",
                    "_predictive_likelihood",
                    "_get_posterior_over_point",
                    "._get_posterior_over_fuzzy_point_in_eval_mode",
                    "_fuzzy_predictive_likelihood",
                },
            },
        )

    def _decorate_class(self, cls: type[ControllerT]) -> type[ControllerT]:
        if issubclass(cls, ClassificationMixin):
            warnings.warn(
                "NormaliseY should not be used above classification decorators "
                "- this may lead to unexpected behaviour.",
                BadCombinationWarning,
                stacklevel=3,
            )

        @wraps_class(cls)
        class InnerClass(cls):
            """
            A wrapper for normalising y inputs and variance.
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                self.rng = utils.optional_random_generator(all_parameters_as_kwargs.pop("rng", None))

                y_std = all_parameters_as_kwargs.pop("y_std")
                train_x = torch.as_tensor(all_parameters_as_kwargs.pop("train_x"))
                train_y = torch.as_tensor(all_parameters_as_kwargs.pop("train_y"))

                if y_std is None:
                    n, *_ = train_x.shape
                    y_std = torch.zeros(n, dtype=self.dtype, device=self.device)

                _normalising_mean, _normalising_std = train_y.float().mean(), train_y.float().std()
                train_y = (train_y - _normalising_mean) / _normalising_std
                y_std = y_std / _normalising_std

                def normalise_posterior_class(posterior_class: type[Posterior]) -> type[Posterior]:
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
                            unscaled_covar = covar * _normalising_std**2
                            return unscaled_mean, unscaled_covar

                        def confidence_interval(
                            self, alpha: float = 0.05
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                            """
                            Un-normalise values.

                            :param alpha: Confidence threshold that interval should cover. The resulting confidence
                                interval should contain the underlying data with probability :math:`1 - \alpha`.
                            :return: The mean and covariance, on the original scale of the data.
                            """
                            mean, lower, upper = super().confidence_interval(alpha)
                            unscaled_mean = (mean * _normalising_std) + _normalising_mean
                            unscaled_lower = (lower * _normalising_std) + _normalising_mean
                            unscaled_upper = (upper * _normalising_std) + _normalising_mean
                            return unscaled_mean, unscaled_lower, unscaled_upper

                        def log_probability(self, y: torch.Tensor) -> torch.Tensor:
                            """
                            Apply the change of variables to the density using the normalise map.

                            :param y: The data-point at which to evaluate the log-probability
                            :return: The log-probability of some data-point ``y``
                            """
                            normalised_y = (y - _normalising_mean) / _normalising_std
                            norm_map_deriv_values = torch.ones_like(y) / _normalising_std
                            jacobian = torch.sum(torch.log(torch.abs(norm_map_deriv_values)))
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

                try:
                    super().__init__(
                        train_x=train_x, train_y=train_y, y_std=y_std, rng=self.rng, **all_parameters_as_kwargs
                    )
                except TypeError as exc:
                    if issubclass(cls, ClassificationMixin):
                        msg = "NormaliseY should not be used above classification decorators."
                        raise TypeError(msg) from exc

            @staticmethod
            def warn_normalise_y():
                """Override base warning because y normalisation has been applied."""

        return InnerClass
