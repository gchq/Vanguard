"""
Contains the multitask_model decorator.
"""
from typing import Type, TypeVar

import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import Mean
from gpytorch.models import GP, ApproximateGP, ExactGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    LMCVariationalStrategy,
    VariationalStrategy,
)
from torch import Tensor

from vanguard.decoratorutils import wraps_class

GPT = TypeVar("GPT", bound=GP)


def multitask_model(cls: Type[GPT]) -> Type[GPT]:
    """
    Convert a model to a multitask model.

    :Example:
        >>> from vanguard.models import ExactGPModel
        >>>
        >>> @multitask_model
        ... class ExactMultitaskModel(ExactGPModel):
        ...     pass
    """
    if issubclass(cls, ExactGP):
        # Pyright does not identify that InnerClass gets renamed
        @wraps_class(cls)
        class InnerClass(cls):  # pyright: ignore [reportRedeclaration]
            """
            A wrapper for applying converting a GP model class to multitask.
            """

            def forward(self, x: Tensor) -> MultitaskMultivariateNormal:  # pyright: ignore [reportIncompatibleMethodOverride]
                """
                Compute the prior latent distribution on a given input.

                .. warning::

                    The signature of this method is incompatible with base class
                    :class:`~gpytorch.likelihoods.multitask_gaussian_likelihood._MultitaskGaussianLikelihoodBase`.

                :param x: (n_samples, n_features) The inputs.
                :returns: The prior distribution.
                """
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    elif issubclass(cls, ApproximateGP):

        @wraps_class(cls)
        class InnerClass(cls):
            """
            A wrapper for applying converting a GP model class to multitask.
            """

            def forward(self, x: Tensor) -> MultivariateNormal:
                """
                Compute the prior latent distribution on a given input.

                :param x: (n_samples, n_features) The inputs.
                :returns: The prior distribution.
                """
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    else:
        raise TypeError(f"Must be applied to a subclass of '{ExactGP.__name__}' or '{ApproximateGP.__name__}'.")

    # Pyright does not detect that wraps_class renames InnerClass
    return InnerClass  # pyright: ignore [reportReturnType]


def independent_variational_multitask_model(cls: Type[GPT]) -> Type[GPT]:
    """Decorate a class to enable independent multitask variational approximation."""

    # Pyright cannot resolve dynamic base class
    @wraps_class(cls)
    class InnerClass(cls):  # pyright: ignore[reportGeneralTypeIssues]
        """
        Implements an independent multitask variational approximation i.e. entirely separate GPs for each task.
        """

        def __init__(
            self,
            train_x: Tensor,
            train_y: Tensor,
            likelihood: GaussianLikelihood,
            mean_module: Mean,
            covar_module: Kernel,
            n_inducing_points: int,
            num_tasks: int,
        ) -> None:
            self.num_tasks = num_tasks
            self.num_latents = self._get_num_latents(mean_module)
            # No suitable base class for ``cls`` to specify this protocol
            super().__init__(train_x, train_y, likelihood, mean_module, covar_module, n_inducing_points)  # pyright: ignore [reportCallIssue]

        def _init_inducing_points(self, train_x: Tensor, n_inducing_points: int) -> Tensor:
            """
            Create the initial inducing points by sampling from the training inputs.

            :param train_x: (n_training_points, n_features)
            :param n_inducing_points: How many inducing points to select.
            :returns: The inducing points sampled from the training points.
            """
            induce_indices = np.random.choice(train_x.shape[0], size=n_inducing_points * self.num_latents, replace=True)
            inducing_points = train_x[induce_indices]
            inducing_points = torch.stack(
                [
                    inducing_points[n_inducing_points * latent_dim : n_inducing_points * (latent_dim + 1)]
                    for latent_dim in range(self.num_latents)
                ]
            )
            return inducing_points

        def _build_variational_strategy(
            self, base_variational_strategy: VariationalStrategy
        ) -> IndependentMultitaskVariationalStrategy:
            return gpytorch.variational.IndependentMultitaskVariationalStrategy(
                base_variational_strategy, num_tasks=self.num_tasks
            )

        def _build_variational_distribution(self, n_inducing_points: int) -> CholeskyVariationalDistribution:
            return gpytorch.variational.CholeskyVariationalDistribution(
                n_inducing_points, batch_shape=torch.Size([self.num_latents])
            )

        def _check_batch_shape(self, mean_module: Mean, covar_module: Kernel) -> None:
            if self.num_tasks == 1:
                raise TypeError(
                    "You are using a multitask variational model in a single-task problem. "
                    "You do not have the correct variational strategy for single"
                    " task. Consider using a single task model instead."
                )

            if self.num_latents != covar_module.batch_shape[-1]:
                raise TypeError(
                    "You are using a multitask variational model but have passed a kernel with batch shape"
                    f"{covar_module.batch_shape}, but a one-dimensional batch shape is required."
                )

            if self.num_tasks != self.num_latents:
                raise ValueError(
                    "You are using a multitask variational model which requires that "
                    "num_tasks==num_latents, but you have supplied mean and kernel with "
                    f"batch_shape {mean_module.batch_shape} whereas num_tasks == {self.num_tasks}."
                    " Possibly you meant to use multitask with LMC?."
                )

        @staticmethod
        def _get_num_latents(mean_module: Mean) -> int:
            """Get the number of latent implied by ``mean_module``."""
            try:
                num_latents = mean_module.batch_shape[-1]
            except IndexError:
                raise TypeError(
                    f"You are using a multitask variational model but have passed a mean with batch shape"
                    f"{mean_module.batch_shape}, but a one-dimensional batch shape is required."
                )
            return num_latents

    # Pyright does not detect that wraps_class renames InnerClass
    return InnerClass  # pyright: ignore [reportReturnType]


def lmc_variational_multitask_model(cls: Type[GPT]) -> Type[GPT]:
    """Decorate a class to enable an LMC multitask variational approximation."""
    new_cls = independent_variational_multitask_model(cls)

    # Pyright cannot resolve dynamic base class
    @wraps_class(new_cls)
    class InnerClass(new_cls):  # pyright: ignore [reportGeneralTypeIssues]
        """
        Implements a linear model of co-regionalisation :cite:`Wackernagel03` multitask variational approximation.
        """

        def _build_variational_strategy(self, base_variational_strategy: VariationalStrategy) -> LMCVariationalStrategy:
            return gpytorch.variational.LMCVariationalStrategy(
                base_variational_strategy, num_tasks=self.num_tasks, num_latents=self.num_latents, latent_dim=-1
            )

        def _check_batch_shape(self, mean_module: Mean, covar_module: Kernel) -> None:
            try:
                # No way to tell Pyright that cls must have been appropriately decorated
                # first
                super()._check_batch_shape(mean_module, covar_module)  # pyright: ignore [reportAttributeAccessIssue]
            except ValueError:
                pass  # num_tasks can differ from num_latents for LMC.

    # Pyright does not detect that wraps_class renames InnerClass
    return InnerClass  # pyright: ignore [reportReturnType]
