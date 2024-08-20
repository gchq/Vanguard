"""
Contains the BaseHierarchicalHyperparameters decorator.
"""

import warnings
from typing import Any, Generator, List, Tuple, Type, TypeVar, Union

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import ScaleKernel
from numpy.typing import NDArray

from vanguard.base import GPController
from vanguard.base.posteriors import MonteCarloPosteriorCollection, Posterior
from vanguard.decoratorutils import Decorator, wraps_class
from vanguard.warnings import _JITTER_WARNING, NumericalWarning

ControllerT = TypeVar("ControllerT", bound=GPController)
DistributionT = TypeVar("DistributionT", bound=gpytorch.distributions.Distribution)
PosteriorT = TypeVar("PosteriorT", bound=Posterior)
ModuleT = TypeVar("ModuleT", bound=gpytorch.module.Module)


class BaseHierarchicalHyperparameters(Decorator):
    """
    Convert a controller so that Bayesian inference is performed over its hyperparameters.

    Note that only those hyperparameters specified using the
    :class:`~vanguard.hierarchical.module.BayesianHyperparameters` decorator will be included
    for Bayesian inference. The remaining hyperparameters will be inferred as point estimates.
    """

    def __init__(self, num_mc_samples: int = 100, **kwargs: Any) -> None:
        """
        Initialise self.

        :param num_mc_samples: The number of Monte Carlo samples to use when approximating
                                    intractable integrals in the variational ELBO and the
                                    predictive posterior.
        """
        self.sample_shape = torch.Size([num_mc_samples])
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def verify_decorated_class(self, cls: Type[ControllerT]) -> None:
        super().verify_decorated_class(cls)
        for previous_decorator in cls.__decorators__:
            if issubclass(previous_decorator, BaseHierarchicalHyperparameters):
                msg = (
                    f"This class is already decorated with `{previous_decorator.__name__}`. "
                    f"Please use only one hierarchical hyperparameters decorator at once."
                )
                raise TypeError(msg)

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        decorator = self

        @wraps_class(cls)
        class InnerClass(cls):
            @classmethod
            def new(cls, instance: Type[ControllerT], **kwargs: Any) -> Type[ControllerT]:
                """Make sure that the hyperparameter collection is copied over."""
                new_instance = super().new(instance, **kwargs)
                new_instance.hyperparameter_collection = instance.hyperparameter_collection
                return new_instance

            def _get_posterior_over_point(self, x: NDArray[np.floating]) -> Type[PosteriorT]:
                """
                Predict the y-value of a single point. The mode (eval vs train) of the model is not changed.

                :param x: (n_predictions, n_features) The predictive inputs.
                :returns: The prior distribution.
                """
                posteriors = (
                    self.posterior_class(posterior_sample)
                    # pylint: disable=protected-access
                    for posterior_sample in decorator._infinite_posterior_samples(self, x)
                )
                posterior_collection = self.posterior_collection_class(posteriors)
                return posterior_collection

            def _predictive_likelihood(self, x: NDArray[np.floating]) -> Type[PosteriorT]:
                """
                Predict the likelihood value of a single point. The mode (eval vs train) of the model is not changed.

                :param x: (n_predictions, n_features) The predictive inputs.
                :returns: The prior distribution.
                """
                likelihoods = (
                    self.posterior_class(posterior_sample)
                    # pylint: disable=protected-access
                    for posterior_sample in decorator._infinite_likelihood_samples(self, x)
                )
                likelihood_collection = self.posterior_collection_class(likelihoods)
                return likelihood_collection

            def _get_posterior_over_fuzzy_point_in_eval_mode(
                self, x: NDArray[np.floating], x_std: Union[NDArray[np.floating], float]
            ) -> Type[MonteCarloPosteriorCollection]:
                """
                Obtain Monte Carlo integration samples from the predictive posterior with Gaussian input noise.

                .. warning:
                    The ``n_features`` must match with :attr:`self.dim`.

                :param x: (n_predictions, n_features) The predictive inputs.
                :param x_std: The input noise standard deviations:

                    * array_like[float]: (n_features,) The standard deviation per input dimension for the predictions,
                    * float: Assume homoskedastic noise.

                :returns: The prior distribution.
                """
                self.set_to_evaluation_mode()
                posteriors = (
                    self.posterior_class(x_sample)
                    # pylint: disable=protected-access
                    for x_sample in decorator._infinite_fuzzy_posterior_samples(self, x, x_std)
                )
                posterior_collection = self.posterior_collection_class(posteriors)
                return posterior_collection

            def _fuzzy_predictive_likelihood(
                self, x: NDArray[np.floating], x_std: Union[NDArray[np.floating], float]
            ) -> Type[MonteCarloPosteriorCollection]:
                """
                Obtain Monte Carlo integration samples from the predictive likelihood with Gaussian input noise.

                .. warning:
                    The ``n_features`` must match with :attr:`self.dim`.

                :param x: (n_predictions, n_features) The predictive inputs.
                :param x_std: The input noise standard deviations:

                    * array_like[float]: (n_features,) The standard deviation per input dimension for the predictions,
                    * float: Assume homoskedastic noise.

                :returns: The prior distribution.
                """
                self.set_to_evaluation_mode()
                likelihoods = (
                    self.posterior_class(posterior_sample)
                    # pylint: disable=protected-access
                    for posterior_sample in decorator._infinite_fuzzy_likelihood_samples(self, x, x_std)
                )
                likelihood_collection = self.posterior_collection_class(likelihoods)
                return likelihood_collection

            def _gp_forward(self, x: torch.Tensor) -> DistributionT:
                """
                Run the forward method of the internal GP model.

                Overloading is necessary to remove fast_pred_var.
                See here: https://github.com/cornellius-gp/gpytorch/issues/864
                """
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=NumericalWarning, message=_JITTER_WARNING)

                    output = self._gp(x)
                return output

        return InnerClass

    @staticmethod
    def _infinite_posterior_samples(
        controller: ControllerT, x: NDArray[np.floating]
    ) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError

    @staticmethod
    def _infinite_fuzzy_posterior_samples(
        controller: ControllerT, x: NDArray[np.floating], x_std: NDArray[np.floating]
    ) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError

    @staticmethod
    def _infinite_likelihood_samples(
        controller: ControllerT, x: NDArray[np.floating]
    ) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError

    @staticmethod
    def _infinite_fuzzy_likelihood_samples(
        controller: ControllerT, x: NDArray[np.floating], x_std: NDArray[np.floating]
    ) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError


def _get_bayesian_hyperparameters(module: ModuleT) -> Tuple[list, ...]:
    """
    Find the bayesian hyperparameters of a GPyTorch module (mean, kernel or likelihood).

    Searches through all sub-modules for parameters and extracts the hyperparameter names,
    the modules to which they belong, their shapes, their constraints and their priors.
    Also finds the ScaleKernels that are not Bayesian (i.e. standard point estimate
    hyperparameters). These are needed to adjust batch_shapes.

    .. note::
        This function is designed to work with modules that have been decorated with
        :class:`~vanguard.hierarchical.module.BayesianHyperparameters`. If that
        decorator has not been applied, then this function does nothing.

    :param module: The module from which to extract the hyperparameters.

    :returns:
        * The module, hyperparameter pairs,
        * The modules (at any depth) corresponding to ScaleKernels with point estimate hyperparameters.
    """
    point_estimates_scale_kernels = []

    bayesian_hyperparameters = getattr(module, "bayesian_hyperparameters", [])
    module_hyperparameter_pairs = [(module, hyperparameter) for hyperparameter in bayesian_hyperparameters]

    for sub_module in module.children():
        sub_hyperparameters, sub_point_estimates_scale_kernels = _get_bayesian_hyperparameters(sub_module)
        module_hyperparameter_pairs.extend(sub_hyperparameters)
        point_estimates_scale_kernels.extend(sub_point_estimates_scale_kernels)

    if isinstance(module, ScaleKernel) and not hasattr(module, "bayesian_hyperparameters"):
        point_estimates_scale_kernels.append(module)

    return module_hyperparameter_pairs, point_estimates_scale_kernels


def extract_bayesian_hyperparameters(controller: ControllerT) -> Tuple[List, List]:
    """Pull hyperparameters and any point-estimate kernels from a controller's mean, kernel and likelihood."""
    hyperparameter_pairs = []

    for module in (controller.mean, controller.likelihood, controller.kernel):
        m_hyperparameters, point_estimate_kernels = _get_bayesian_hyperparameters(module)
        hyperparameter_pairs.extend(m_hyperparameters)
    return hyperparameter_pairs, point_estimate_kernels


def set_batch_shape(kwargs: Any, module_name: str, batch_shape: Tuple[int, ...]) -> None:
    """Set the batch shape in kwargs dictionary which may not exist."""
    kwargs_name = f"{module_name}_kwargs"
    module_kwargs = kwargs.pop(kwargs_name, {})
    module_kwargs["batch_shape"] = batch_shape
    kwargs[kwargs_name] = module_kwargs
