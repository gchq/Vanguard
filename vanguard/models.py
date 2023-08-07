"""
Vanguard implements a small number of base models which are built on by various decorators.

They are syntactically similar to the standard model classes used in GPyTorch.
"""
import gpytorch
from gpytorch.models import ExactGP
import numpy as np


class ExactGPModel(ExactGP):
    """
    Standard GPyTorch exact GP model subclassing :py:class:`gpytorch.models.ExactGP` with flexible prior kernel, mean.
    """
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module, **kwargs):
        """
        Initialise self.

        :param torch.Tensor train_x: (n_samples, n_features) The training inputs (features).
        :param torch.Tensor train_y: (n_samples,) The training targets (response).
        :param gpytorch.likelihoods.GaussianLikelihood likelihood:  Likelihood to use with model.
                Since we're using exact inference, the likelihood must be Gaussian.
        :param gpytorch.means.Mean mean_module: The prior mean function to use.
        :param gpytorch.kernels.Kernel covar_module:  The prior kernel function to use.
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        """
        Compute the prior latent distribution on a given input.

        :param torch.Tensor x: (n_samples, n_features) The inputs.
        :returns: The prior distribution.
        :rtype: gpytorch.distributions.MultivariateNormal
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class InducingPointKernelGPModel(ExactGPModel):
    """
    A model with inducing point sparse approximation to the kernel.

    GPyTorch exact GP model subclassing :py:class:`gpytorch.models.ExactGP` with flexible prior kernel, mean and an
    inducing point sparse approximation to the kernel a la [Titsias09]_.
    """
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module, n_inducing_points):
        """
        Initialise self.

        :param torch.Tensor train_x: (n_samples, n_features) The training inputs (features).
        :param torch.Tensor train_y: (n_samples,) The training targets (response).
        :param gpytorch.likelihoods.GaussianLikelihood likelihood:  Likelihood to use with model.
                Since we're using exact inference, the likelihood must be Gaussian.
        :param gpytorch.means.Mean mean_module: The prior mean function to use.
        :param gpytorch.kernels.Kernel covar_module:  The prior kernel function to use.
        :param int n_inducing_points: The number of inducing points in the sparse kernel approximation.
        """
        inducing_point_indices = np.random.choice(train_x.shape[0], size=n_inducing_points, replace=True)
        inducing_points = train_x[inducing_point_indices, :].clone()
        covar_module = gpytorch.kernels.InducingPointKernel(covar_module, inducing_points=inducing_points,
                                                            likelihood=likelihood)
        super().__init__(train_x, train_y, likelihood, mean_module, covar_module)
