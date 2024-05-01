"""
The :class:`GaussianGPController` provides the user with a standard GP model with no extra features.
"""
from typing import Type, Union

import gpytorch
import numpy as np
import numpy.typing
import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import GPController
from .optimise import GreedySmartOptimiser
from .optimise.optimiser import SmartOptimiser


class GaussianGPController(GPController):
    """
    Base class for implementing standard GP regression with flexible prior kernel and mean functions.

    This is the best starting point for users, containing many sensible default values.
    The standard reference is :cite:`Rasmussen06`.
    """
    def __init__(self, train_x: numpy.typing.NDArray[np.floating], train_y: numpy.typing.NDArray[np.floating],
                 kernel_class: Type[gpytorch.kernels.Kernel], y_std: Union[numpy.typing.NDArray[np.floating], float],
                 mean_class: Type[gpytorch.means.Mean] = ConstantMean,
                 likelihood_class: Type[gpytorch.likelihoods.Likelihood] = FixedNoiseGaussianLikelihood,
                 marginal_log_likelihood_class: Type[gpytorch.mlls.MarginalLogLikelihood] = ExactMarginalLogLikelihood,
                 optimiser_class: Type[torch.optim.Optimizer] = torch.optim.Adam, smart_optimiser_class: Type[SmartOptimiser] = GreedySmartOptimiser,
                 **kwargs):
        """
        Initialise self.

        :param train_x: (n_samples, n_features) The inputs (or the observed values)
        :param train_y: (n_samples,) or (n_samples, 1) The responsive values.
        :param kernel_class: An uninstantiated subclass of :class:`gpytorch.kernels.Kernel`.
        :param mean_class: An uninstantiated subclass of :class:`gpytorch.means.Mean` to use in the prior GP.
                Defaults to :class:`gpytorch.means.ConstantMean`.
        :param y_std: The observation noise standard deviation:

            * *array_like[float]* (n_samples,): known heteroskedastic noise,
            * *float*: known homoskedastic noise assumed.

        :param likelihood_class: An uninstantiated subclass of :class:`gpytorch.likelihoods.Likelihood`.
                The default is :class:`gpytorch.likelihoods.FixedNoiseGaussianLikelihood`.
        :param marginal_log_likelihood_class: An uninstantiated subclass of of an MLL from
                :mod:`gpytorch.mlls`. The default is :class:`gpytorch.mlls.ExactMarginalLogLikelihood`.
        :param optimiser_class: An uninstantiated :class:`torch.optim.Optimizer` class used for
                gradient-based learning of hyperparameters. The default is :class:`torch.optim.Adam`.
        :param kwargs: For a complete list, see :class:`~vanguard.base.gpcontroller.GPController`.
        """
        super().__init__(train_x=train_x, train_y=train_y, kernel_class=kernel_class, mean_class=mean_class,
                         y_std=y_std, likelihood_class=likelihood_class,
                         marginal_log_likelihood_class=marginal_log_likelihood_class, optimiser_class=optimiser_class,
                         smart_optimiser_class=smart_optimiser_class, **kwargs)
