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

"""Implementation of tempered Laplace approximation approach to Bayesian hyperparameters."""

import itertools
from collections.abc import Generator
from math import ceil
from typing import Any, Callable, Optional, TypeVar, Union

import gpytorch
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import Self, override

from vanguard import utils
from vanguard.decoratorutils import process_args, wraps_class
from vanguard.hierarchical.base import (
    BaseHierarchicalHyperparameters,
    GPController,
    Posterior,
    extract_bayesian_hyperparameters,
    set_batch_shape,
)
from vanguard.hierarchical.collection import OnePointHyperparameterCollection
from vanguard.hierarchical.distributions import SpectralRegularisedMultivariateNormal
from vanguard.variational import VariationalInference

HESSIAN_JITTER = 1e-5

ControllerT = TypeVar("ControllerT", bound=GPController)
LikelihoodT = TypeVar("LikelihoodT", bound=gpytorch.likelihoods.GaussianLikelihood)
PosteriorT = TypeVar("PosteriorT", bound=Posterior)
# pylint: disable-next=protected-access
VariationalDistributionT = TypeVar("VariationalDistributionT", bound=gpytorch.variational._VariationalDistribution)


class LaplaceHierarchicalHyperparameters(BaseHierarchicalHyperparameters):
    """
    Convert a controller so that Bayesian inference is performed over its hyperparameters.

    A post-hoc Laplace approximation is to obtain an approximation hyperparameter posterior.
    Note that only those hyperparameters specified using the
    :class:`~vanguard.hierarchical.module.BayesianHyperparameters` decorator will be included
    for Bayesian inference. The remaining hyperparameters will be inferred as point estimates.

    :Example:
        >>> from gpytorch.kernels import RBFKernel
        >>> import numpy as np
        >>> import torch
        >>> from vanguard.vanilla import GaussianGPController
        >>> from vanguard.hierarchical import (BayesianHyperparameters,
        ...                                    LaplaceHierarchicalHyperparameters)
        >>>
        >>> @LaplaceHierarchicalHyperparameters(num_mc_samples=50)
        ... class HierarchicalController(GaussianGPController):
        ...     pass
        >>>
        >>> @BayesianHyperparameters()
        ... class BayesianRBFKernel(RBFKernel):
        ...     pass
        >>>
        >>> train_x = torch.tensor([0, 0.5, 0.9, 1])
        >>> rng = torch.Generator(device=train_x.device).manual_seed(1234)
        >>> train_y = torch.normal(mean=1 / (1 + train_x), std=torch.ones_like(train_x) * 0.005, generator=rng)
        >>> gp = HierarchicalController(train_x, train_y, BayesianRBFKernel, y_std=0.0)
        >>> loss = gp.fit(100)
        >>>
        >>> test_x = torch.tensor([0.05, 0.95])
        >>> mean, lower, upper = gp.posterior_over_point(test_x).confidence_interval()
        >>> (upper > 1/(1 + test_x)).all().item(), (lower < 1/(1 + test_x)).all().item()
        (True, True)
    """

    def __init__(
        self, num_mc_samples: int = 100, temperature: Optional[float] = None, uv_cutoff: float = 1e-3, **kwargs: Any
    ) -> None:
        """
        Initialise self.

        :param num_mc_samples: The number of Monte Carlo samples to use when approximating
                                    intractable integrals in the variational ELBO and the
                                    predictive posterior.
        :param temperature: The (inverse) scale for tempering the posterior, for balancing
                                    exploration and exploitation of the target distribution.
                                    If :data:`None`, it's set automatically using a trace rescaling heuristic.
        :param uv_cutoff: The cutoff for eigenvalues in computing the eigenbasis and spectrum
                                    of the Hessian. For eigenvalues below this cutoff, the Hessian
                                    inverse eigenvalues are set to a fixed small jitter value.
        :param kwargs: Keyword arguments passed to :py:class:`~vanguard.decoratorutils.basedecorator.Decorator`.
        """
        super().__init__(num_mc_samples=num_mc_samples, **kwargs)
        self.temperature = temperature
        self.uv_cutoff = uv_cutoff

    @property
    @override
    def safe_updates(self) -> dict[type, set[str]]:
        return self._add_to_safe_updates(
            super().safe_updates,
            {VariationalInference: {"__init__", "_predictive_likelihood", "_fuzzy_predictive_likelihood"}},
        )

    def _decorate_class(self, cls: type[ControllerT]) -> type[ControllerT]:
        uv_cutoff = self.uv_cutoff
        posterior_temperature = self.temperature
        base_decorated_cls = super()._decorate_class(cls)

        @wraps_class(base_decorated_cls, decorator_source=self)
        class InnerClass(base_decorated_cls):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                for module_name in ("kernel", "mean", "likelihood"):
                    set_batch_shape(kwargs, module_name, torch.Size([]))

                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                self.rng = utils.optional_random_generator(all_parameters_as_kwargs.pop("rng", None))
                # Pop `rng` from kwargs to ensure we don't provide duplicate values to superclass init
                kwargs.pop("rng", None)

                super().__init__(*args, rng=self.rng, **kwargs)

                module_hyperparameter_pairs, _ = extract_bayesian_hyperparameters(self)

                self.hyperparameter_collection = OnePointHyperparameterCollection(module_hyperparameter_pairs)

                self._smart_optimiser.update_registered_module(self._gp)
                mean = torch.zeros(self.hyperparameter_collection.hyperparameter_dimension)
                cov_evals = torch.ones(self.hyperparameter_collection.hyperparameter_dimension)
                cov_evecs = torch.eye(self.hyperparameter_collection.hyperparameter_dimension)
                self.hyperparameter_posterior = torch.distributions.MultivariateNormal(
                    loc=mean, covariance_matrix=cov_evecs
                )
                self.hyperparameter_posterior_mean = mean
                self.hyperparameter_posterior_covariance = cov_evals, cov_evecs
                self._temperature = posterior_temperature

            @classmethod
            def new(cls, instance: Self, **kwargs: Any) -> Self:
                """Copy hyperparameter posteriors."""
                new_instance = super().new(instance, **kwargs)
                new_instance.hyperparameter_posterior_mean = (
                    instance.hyperparameter_posterior_mean  # pyright: ignore[reportAttributeAccessIssue]
                )
                new_instance.hyperparameter_posterior_covariance = (
                    instance.hyperparameter_posterior_covariance  # pyright: ignore[reportAttributeAccessIssue]
                )
                new_instance.temperature = instance.temperature  # pyright: ignore[reportAttributeAccessIssue]
                return new_instance

            @property
            def temperature(self) -> Optional[float]:
                return self._temperature

            @temperature.setter
            def temperature(self, value: Optional[float]) -> None:
                self._temperature = value
                self._update_hyperparameter_posterior()

            def _sgd_round(self, *args: Any, **kwargs: Any) -> float:
                loss = super()._sgd_round(*args, **kwargs)

                posterior_params = self._compute_hyperparameter_laplace_approximation()
                self.hyperparameter_posterior_mean, self.hyperparameter_posterior_covariance = posterior_params
                if self.temperature is None:
                    self.temperature = self.auto_temperature()
                else:
                    self._update_hyperparameter_posterior()
                return loss

            def _compute_hyperparameter_laplace_approximation(
                self,
            ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                hessian = self._compute_loss_hessian().detach().clone()
                eigenvalues, eigenvectors = _subspace_hessian_inverse_eig(hessian, cutoff=uv_cutoff)
                mean = self.hyperparameter_collection.hyperparameter_tensor
                return mean, (eigenvalues.detach().clone(), eigenvectors.detach().clone())

            def _compute_loss_hessian(self) -> torch.Tensor:
                batch_size = self.batch_size if self.batch_size else len(self.train_x)
                single_epoch_iters = ceil(len(self.train_x) / batch_size)

                total_loss = 0
                for train_x, train_y, train_y_noise in itertools.islice(self.train_data_generator, single_epoch_iters):
                    # Pylint false positive here - this should be defined in the parent class
                    self.likelihood_noise = train_y_noise  # pylint: disable=attribute-defined-outside-init
                    total_loss += self._loss(train_x, train_y)

                gradient_list = torch.autograd.grad(total_loss, iter(self.hyperparameter_collection), create_graph=True)
                gradients = torch.cat([grad.reshape(-1) for grad in gradient_list])
                hessian_dimension = self.hyperparameter_collection.hyperparameter_tensor.shape[0]
                hessian = torch.zeros(hessian_dimension, hessian_dimension)

                for index, gradient in enumerate(gradients):
                    sub_gradient_list = torch.autograd.grad(
                        gradient, iter(self.hyperparameter_collection), create_graph=True
                    )
                    sub_gradients = torch.cat([grad.reshape(-1) for grad in sub_gradient_list])
                    hessian[index] = sub_gradients
                return hessian

            def _sample_and_set_hyperparameters(self) -> None:
                sample = self.hyperparameter_posterior.rsample()
                self.hyperparameter_collection.hyperparameter_tensor = sample

            def _update_hyperparameter_posterior(self) -> None:
                """Set the hyperparameter posterior distribution using the current parameters."""
                mean = self.hyperparameter_posterior_mean
                eigenvalues, eigenvectors = self.hyperparameter_posterior_covariance
                new_eigenvalues = eigenvalues * self.temperature
                laplace_distribution = SpectralRegularisedMultivariateNormal.from_eigendecomposition(
                    mean, new_eigenvalues, eigenvectors
                )
                self.hyperparameter_posterior = laplace_distribution

            def auto_temperature(self) -> float:
                """Set the temperature automatically using a trace rescaling heuristic."""
                return 1 / torch.sum(self.hyperparameter_posterior_covariance[0]).item()

        return InnerClass

    @staticmethod
    def _infinite_posterior_samples(
        controller: ControllerT, x: Union[Tensor, NDArray[np.floating]]
    ) -> Generator[torch.Tensor, None, None]:
        """
        Yield posterior samples forever.

        :param controller: The controller from which to yield samples.
        :param x: (n_predictions, n_features) The predictive inputs.
        """
        tx = torch.as_tensor(x, dtype=torch.float32, device=controller.device)
        while True:
            # pylint: disable-next=protected-access
            controller._sample_and_set_hyperparameters()
            # pylint: disable-next=protected-access
            yield controller._gp_forward(tx).add_jitter(1e-3)

    @staticmethod
    def _infinite_fuzzy_posterior_samples(
        controller: ControllerT,
        x: Union[Tensor, NDArray[np.floating]],
        x_std: Union[Tensor, NDArray[np.floating], float],
    ) -> Generator[PosteriorT, None, None]:
        """
        Yield fuzzy posterior samples forever.

        :param controller: The controller from which to yield samples.
        :param x: (n_predictions, n_features) The predictive inputs.
        :param x_std: The input noise standard deviations:

            * array_like[float]: (n_features,) The standard deviation per input dimension for the predictions,
            * float: Assume homoskedastic noise.

        :return: Generator that provides posterior samples.
        """
        tx = torch.tensor(x, dtype=torch.float32, device=controller.device)
        tx_std = controller._process_x_std(x_std).to(controller.device)  # pylint: disable=protected-access
        while True:
            # pylint: disable-next=protected-access
            controller._sample_and_set_hyperparameters()  # type: ignore[reportAttributeAccessIssue]
            sample_shape = x.shape
            x_sample = torch.randn(size=sample_shape, device=controller.device) * tx_std + tx
            # pylint: disable-next=protected-access
            output = controller._gp_forward(x_sample).add_jitter(1e-3)
            yield output

    @staticmethod
    def _infinite_likelihood_samples(
        controller: ControllerT, x: Union[Tensor, NDArray[np.floating]]
    ) -> Generator[PosteriorT, None, None]:
        """
        Yield likelihood samples forever.

        :param controller: The controller from which to yield samples.
        :param x: (n_predictions, n_features) The predictive inputs.
        :return: Generator that provides likelihood samples.
        """
        func = _posterior_to_likelihood_samples(LaplaceHierarchicalHyperparameters._infinite_posterior_samples)
        yield from func(controller, x)

    @staticmethod
    def _infinite_fuzzy_likelihood_samples(
        controller: ControllerT,
        x: Union[Tensor, NDArray[np.floating]],
        x_std: Union[Tensor, NDArray[np.floating], float],
    ) -> Generator[torch.Tensor, None, None]:
        """
        Yield fuzzy likelihood samples forever.

        :param controller: The controller from which to yield samples.
        :param x: (n_predictions, n_features) The predictive inputs.
        :param x_std: The input noise standard deviations:

            * array_like[float]: (n_features,) The standard deviation per input dimension for the predictions,
            * float: Assume homoskedastic noise.

        :return: Generator that provides likelihood samples.
        """
        func = _posterior_to_likelihood_samples(LaplaceHierarchicalHyperparameters._infinite_fuzzy_posterior_samples)
        # TODO: x_std was previously unused, but this function failed when writing unit tests.
        #  Is passing x_std below the correct behaviour?
        # https://github.com/gchq/Vanguard/issues/301
        yield from func(controller, x, x_std)


def _subspace_hessian_inverse_eig(hessian: torch.Tensor, cutoff: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a sort-of-inverse of the Hessian and return its eigenbasis and spectrum.

    Its spectrum is deformed to effectively project-out its 'bad' directions.
    'Bad' means negative or very small and positive.
    Negative strictly break the Laplace approximation, so we must remove them.
    Small eigenvalues correspond to very flat directions along which the truncated
    Taylor expansion behind the Laplace approximation breaks down.
    Along bad directions, we set the Hessian inverse eigenvalues to a fixed
    small jitter value.

    :param hessian: Hessian matrix we wish to invert
    :param cutoff: Eigenvalues smaller than `cutoff` will be discarded from computations
    :return: Arrays holding inverse_eigenvalues and eigenvectors
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)  # pylint: disable=not-callable
    keep_indices = eigenvalues > cutoff
    inverse_eigenvalues = 1 / eigenvalues
    inverse_eigenvalues[~keep_indices] = HESSIAN_JITTER
    return inverse_eigenvalues, eigenvectors


def _posterior_to_likelihood_samples(
    posterior_generator: Callable[[ControllerT, NDArray[np.floating]], Generator[torch.Tensor, None, None]],
) -> Callable[[ControllerT, NDArray[np.floating]], Generator[torch.Tensor, None, None]]:
    """
    Convert an infinite posterior sample generator to generate likelihood samples.

    :param posterior_generator: Generator objective that provides posterior objects
    :return: Generator object that provides likelihood samples.
    """

    def generator(
        controller: ControllerT, x: Union[Tensor, NDArray[np.floating]], *args
    ) -> Generator[Tensor, None, None]:
        """
        Yield likelihood samples forever.

        :param controller: The controller from which to yield samples.
        :param x: (n_predictions, n_features) The predictive inputs.
        :return: Generator that provides likelihood samples.
        """
        for sample in posterior_generator(controller, x, *args):
            # pylint: disable-next=protected-access
            shape = controller._decide_noise_shape(controller.posterior_class(sample), x)
            noise = torch.zeros(shape, dtype=torch.float32, device=controller.device)
            # pylint: disable-next=protected-access
            likelihood_output = controller._likelihood(sample, noise=noise)
            yield likelihood_output

    return generator
