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
Enable variational inference in a controller.

The :class:`VariationalInference` decorator primes a :class:`~vanguard.base.gpcontroller.GPController` class
for variational inference.
"""

from typing import Any, Generic, Optional, TypeVar, Union

import gpytorch.settings
import numpy as np
import numpy.typing
import torch
from torch import Tensor

from vanguard import utils
from vanguard.base import GPController
from vanguard.base.posteriors import Posterior
from vanguard.decoratorutils import Decorator, process_args, wraps_class
from vanguard.variational.models import SVGPModel

ControllerT = TypeVar("ControllerT", bound=GPController)
# pylint: disable-next=protected-access
StrategyT = TypeVar("StrategyT", bound=gpytorch.variational._VariationalStrategy)
# pylint: disable-next=protected-access
DistributionT = TypeVar("DistributionT", bound=gpytorch.variational._VariationalDistribution)


class VariationalInference(Decorator, Generic[StrategyT, DistributionT]):
    """
    Set-up a :class:`~vanguard.base.gpcontroller.GPController` class for variational inference.

    This is best used when:

    * the posterior can not be calculated as a closed-form, or
    * there are too many points to train a model in a reasonable time (see :cite:`Cheng17`).

    .. note::
        This decorator does not take the standard parameters in the
        :class:`~vanguard.decoratorutils.basedecorator.Decorator`
        class, as it only affects the input.

    .. warning::
        This decorator will force the wrapped controller class to only accept compatible
        ``gp_model_class`` and ``marginal_log_likelihood_class`` arguments. The former should
        be a subclass of :class:`vanguard.variational.models.SVGPModel`, and the latter must take a ``num_data``
        :class:`int` argument (e.g. a subclass of :ref:`one of the following
        </marginal_log_likelihoods.rst#approximate-gp-inference>`).

    :Example:
        >>> @VariationalInference(n_inducing_points=100)
        ... class NewController(GPController):
        ...     pass
    """

    def __init__(
        self,
        n_inducing_points: Optional[int] = None,
        n_likelihood_samples: int = 10,
        variational_strategy_class: Optional[type[StrategyT]] = None,
        variational_distribution_class: Optional[type[DistributionT]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise self.

        :param n_inducing_points: The size of the inducing point approximation. Defaults to None, meaning
                                           that the number of inducing points will be set to the number of points.
        :param n_likelihood_samples: If the marginal likelihood cannot be computed exactly (which is usually the
                                         case when using variational inference), it is approximated using
                                         MC integration by sampling from the variational posterior and averaging over
                                         the likelihood values for each sample. This is the number of samples to use.
        :param variational_strategy_class: The class for the variational strategy to use.
                                                     Default behaviour is defined in
                                                     :class:`gpytorch.variational.VariationalStrategy`
                                                     (:cite:`Hensman15`).
        :param variational_distribution_class: The class for the variational distribution to use.
            Default behaviour is defined in
            :class:`gpytorch.variational.CholeskyVariationalDistribution` (Cholesky).
        """
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)
        self.n_inducing_points = n_inducing_points
        self.n_likelihood_samples = n_likelihood_samples
        self.variational_strategy_class = variational_strategy_class
        self.gp_model_class = self._build_gp_model_class(variational_distribution_class, variational_strategy_class)

    def _build_gp_model_class(
        self,
        variational_distribution_class: Optional[type[DistributionT]],
        variational_strategy_class: Optional[type[StrategyT]],
    ) -> type[SVGPModel]:
        if variational_distribution_class is not None:

            @wraps_class(SVGPModel)
            class VDistGPModel(SVGPModel):
                def _build_variational_distribution(self, n_inducing_points: int) -> DistributionT:
                    return variational_distribution_class(n_inducing_points)
        else:

            @wraps_class(SVGPModel)
            class VDistGPModel(SVGPModel):
                pass

        if variational_strategy_class is not None:
            variational_strategy_class.approximation_size = self.n_inducing_points

            @wraps_class(VDistGPModel)
            class NewGPModel(VDistGPModel):
                def _build_base_variational_strategy(
                    self, inducing_points: Tensor, variational_distribution: DistributionT
                ) -> StrategyT:
                    return variational_strategy_class(self, inducing_points, variational_distribution)
        else:

            @wraps_class(VDistGPModel)
            class NewGPModel(VDistGPModel):
                pass

        return NewGPModel

    def _decorate_class(self, cls: type[ControllerT]) -> type[ControllerT]:
        n_inducing_points = self.n_inducing_points
        decorator = self
        _gp_model_class = self.gp_model_class

        @wraps_class(cls, decorator_source=self)
        class InnerClass(cls):
            """
            A wrapper for implementing variational inference.
            """

            gp_model_class = _gp_model_class

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)

                self.rng = utils.optional_random_generator(all_parameters_as_kwargs.pop("rng", None))

                train_x = torch.as_tensor(all_parameters_as_kwargs.pop("train_x"))
                train_y = torch.as_tensor(all_parameters_as_kwargs.pop("train_y"))

                gp_kwargs = all_parameters_as_kwargs.pop("gp_kwargs", {})
                gp_kwargs["n_inducing_points"] = n_inducing_points or train_x.shape[0]

                mll_kwargs = all_parameters_as_kwargs.pop("mll_kwargs", {})
                mll_kwargs["num_data"] = train_y.numel()

                try:
                    super().__init__(
                        train_x=train_x,
                        train_y=train_y,
                        gp_kwargs=gp_kwargs,
                        mll_kwargs=mll_kwargs,
                        rng=self.rng,
                        **all_parameters_as_kwargs,
                    )
                except TypeError as error:
                    if "__init__() got an unexpected keyword argument 'num_data'" in str(error):
                        raise TypeError(
                            "The class passed to `marginal_log_likelihood_class` must take a "
                            "`num_data: int` argument, since we run variational inference with SGD."
                        ) from error
                    else:
                        raise

            def _predictive_likelihood(self, x: Union[numpy.typing.NDArray[np.floating], float]) -> Posterior:
                with gpytorch.settings.num_likelihood_samples(decorator.n_likelihood_samples):
                    return super()._predictive_likelihood(x)

            def _fuzzy_predictive_likelihood(
                self,
                x: Union[numpy.typing.NDArray[np.floating], float],
                x_std: Union[numpy.typing.NDArray[np.floating], float],
            ) -> Posterior:
                with gpytorch.settings.num_likelihood_samples(decorator.n_likelihood_samples):
                    return super()._fuzzy_predictive_likelihood(x, x_std)

        # Ignore type errors here - static type checkers don't understand that we dynamically inherit from `cls`, so
        # `InnerClass` is always a subtype of `cls`
        return InnerClass  # pyright: ignore[reportReturnType]
