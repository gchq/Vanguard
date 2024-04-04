"""
Enable variational inference in a controller.

The class:`VariationalInference` decorator primes a class:`~vanguard.base.gpcontroller.GPController` class
for variational inference.
"""
from typing import Optional, Type, TypeVar, Union

import gpytorch.settings
import numpy as np
import numpy.typing

from ..base import GPController
from ..base.posteriors import Posterior
from ..decoratorutils import Decorator, process_args, wraps_class
from .models import SVGPModel


ControllerT = TypeVar("ControllerT", bound=GPController)
class VariationalInference(Decorator):
    """
    Set-up a class:`~vanguard.base.gpcontroller.GPController` class for variational inference.

    This is best used when:

        * the posterior can not be calculated as a closed-form, or
        * there are too many points to train a model in a reasonable time (see :cite:`Cheng17`).

    .. note::
        This decorator does not take the standard parameters in the
        class:`~vanguard.decoratorutils.basedecorator.Decorator`
        class, as it only affects the input.

    .. warning::
        This decorator will force the wrapped controller class to only accept compatible
        ``gp_model_class`` and ``marginal_log_likelihood_class`` arguments. The former should
        be a subclass of class:`vanguard.variational.models.SVGPModel`, and the latter must take a `num_data` int
        argument (e.g. a subclass of class:`gpytorch.mlls._ApproximateMarginalLogLikelihood`).

    :Example:
        >>> @VariationalInference(n_inducing_points=100)
        ... class NewController(GPController):
        ...     pass
    """
    def __init__(
         self,
         n_inducing_points: Optional[int] = None,
         n_likelihood_samples: int = 10,
         variational_strategy_class: Optional[Type[gpytorch.variational._VariationalStrategy]] = None,
         variational_distribution_class: Optional[Type[gpytorch.variational._VariationalDistribution]] = None,
         **kwargs
    ):
        """
        Initialise self.

        :param int,None n_inducing_points: The size of the inducing point approximation. Defaults to None, meaning
                                           that the number of inducing points will be set to the number of points.
        :param int n_likelihood_samples: If the marginal likelihood cannot be computed exactly (which is usually the
                                         case when using variational inference), it is approximated using
                                         MC integration by sampling from the variational posterior and averaging over
                                         the likelihood values for each sample. This is the number of samples to use.
        :param type,None variational_strategy_class: The class for the variational strategy to use.
                                                     Default behaviour is defined in
                                                     class:`gpytorch.variational.VariationalStrategy`
                                                     (:cite:`Hensman15`).
        :param type,None variational_distribution_class: The class for the variational distribution to use.
            Default behaviour is defined in
            class:`gpytorch.variational.CholeskyVariationalDistribution` (Cholesky).
        """
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)
        self.n_inducing_points = n_inducing_points
        self.n_likelihood_samples = n_likelihood_samples
        self.variational_strategy_class = variational_strategy_class
        self.gp_model_class = self._build_gp_model_class(variational_distribution_class, variational_strategy_class)

    def _build_gp_model_class(
        self,
        variational_distribution_class: Optional[Type[gpytorch.variational._VariationalDistribution]],
        variational_strategy_class: Optional[Type[gpytorch.variational._VariationalStrategy]],
    ) -> Type[SVGPModel]:
        if variational_distribution_class is not None:
            @wraps_class(SVGPModel)
            class VDistGPModel(SVGPModel):
                def _build_variational_distribution(self, n_inducing_points):
                    return variational_distribution_class(n_inducing_points)
        else:
            @wraps_class(SVGPModel)
            class VDistGPModel(SVGPModel):
                pass
        if variational_strategy_class is not None:
            variational_strategy_class.approximation_size = self.n_inducing_points
            @wraps_class(VDistGPModel)
            class NewGPModel(VDistGPModel):
                def _build_base_variational_strategy(self, inducing_points, variational_distribution):
                    return variational_strategy_class(self, inducing_points, variational_distribution)
        else:
            @wraps_class(VDistGPModel)
            class NewGPModel(VDistGPModel):
                pass
        return NewGPModel

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        n_inducing_points = self.n_inducing_points
        decorator = self
        _gp_model_class = self.gp_model_class

        @wraps_class(cls)
        class InnerClass(cls):
            """
            A wrapper for implementing variational inference.
            """
            gp_model_class = _gp_model_class

            def __init__(self, *args, **kwargs):
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                all_parameters_as_kwargs.pop("self")

                train_x = all_parameters_as_kwargs.pop("train_x")
                train_y = all_parameters_as_kwargs.pop("train_y")

                gp_kwargs = all_parameters_as_kwargs.pop("gp_kwargs", {})
                gp_kwargs["n_inducing_points"] = n_inducing_points or train_x.shape[0]

                mll_kwargs = all_parameters_as_kwargs.pop("mll_kwargs", {})
                mll_kwargs["num_data"] = train_y.size

                try:
                    super().__init__(train_x=train_x, train_y=train_y, gp_kwargs=gp_kwargs, mll_kwargs=mll_kwargs,
                                     **all_parameters_as_kwargs)
                except TypeError as error:
                    if str(error) == "__init__() got an unexpected keyword argument 'num_data'":
                        raise ValueError("The class passed to ``marginal_log_likelihood_class`` must take a "
                                         "``num_data`` class:`int` argument since we run "
                                         "variational inference with SGD.")
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

        # ignore type errors here - static type checkers don't understand that we dynamically inherit from `cls`, so
        # `InnerClass` is always a subtype of `cls`
        return InnerClass  # type: ignore
