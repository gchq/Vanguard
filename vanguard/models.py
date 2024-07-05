"""
Vanguard implements a small number of base models which are built on by various decorators.

They are syntactically similar to the standard model classes used in GPyTorch.
"""

from typing import Any, Optional

import gpytorch
import numpy as np
import torch
from gpytorch.models import ExactGP

from vanguard import utils


class ExactGPModel(ExactGP):
    """
    Standard GPyTorch exact GP model subclassing :class:`gpytorch.models.ExactGP` with flexible prior kernel, mean.
    """

    def __init__(  # pylint: disable=unused-argument
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        **kwargs: Any,
    ) -> None:
        """
        Initialise self.

        :param train_x: (n_samples, n_features) The training inputs (features).
        :param train_y: (n_samples,) The training targets (response).
        :param likelihood: Likelihood to use with model. Since we're using exact inference, the likelihood must
            be Gaussian.
        :param mean_module: The prior mean function to use.
        :param covar_module: The prior kernel function to use.
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module
        # TODO: warn if kwargs is non-empty here?
        # https://github.com/gchq/Vanguard/issues/219

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:  # pylint: disable=arguments-differ
        """
        Compute the prior latent distribution on a given input.

        :param x: (n_samples, n_features) The inputs.
        :returns: The prior distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class InducingPointKernelGPModel(ExactGPModel):
    """
    A model with inducing point sparse approximation to the kernel.

    GPyTorch exact GP model subclassing :class:`gpytorch.models.ExactGP` with flexible prior kernel, mean and an
    inducing point sparse approximation to the kernel a la :cite:`Titsias09`.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        n_inducing_points: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Initialise self.

        :param train_x: (n_samples, n_features) The training inputs (features).
        :param train_y: (n_samples,) The training targets (response).
        :param likelihood: Likelihood to use with model. Since we're using exact inference, the likelihood must
            be Gaussian.
        :param mean_module: The prior mean function to use.
        :param covar_module: The prior kernel function to use.
        :param n_inducing_points: The number of inducing points in the sparse kernel approximation.
        :param rng: Generator instance used to generate random numbers.
        """
        rng = utils.optional_random_generator(rng)
        inducing_point_indices = rng.choice(train_x.shape[0], size=n_inducing_points, replace=True)
        inducing_points = train_x[inducing_point_indices, :].clone()
        covar_module = gpytorch.kernels.InducingPointKernel(
            covar_module, inducing_points=inducing_points, likelihood=likelihood
        )
        super().__init__(train_x, train_y, likelihood, mean_module, covar_module)
