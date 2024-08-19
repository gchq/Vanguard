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
The :class:`GaussianGPController` provides the user with a standard GP model with no extra features.
"""

from typing import Any, Optional, Type, Union

import gpytorch
import numpy as np
import numpy.typing
import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from vanguard import utils
from vanguard.base import GPController
from vanguard.optimise import GreedySmartOptimiser
from vanguard.optimise.optimiser import SmartOptimiser


class GaussianGPController(GPController):
    """
    Base class for implementing standard GP regression with flexible prior kernel and mean functions.

    This is the best starting point for users, containing many sensible default values.
    The standard reference is :cite:`Rasmussen06`.

    :param train_x: (n_samples, n_features) The inputs (or the observed values).
    :param train_y: (n_samples,) or (n_samples, 1) The responsive values.
    :param kernel_class: An uninstantiated subclass of :class:`gpytorch.kernels.Kernel`.
    :param y_std: The observation noise standard deviation, one of:

        * :class:`~numpy.ndarray` (n_samples,): known heteroskedastic noise.
        * :class:`float`: known homoskedastic noise assumed.

    :param mean_class: An uninstantiated subclass of :class:`gpytorch.means.Mean` to use in the prior GP.
        Defaults to :class:`gpytorch.means.ConstantMean`.
    :param likelihood_class: An uninstantiated subclass of :class:`gpytorch.likelihoods.Likelihood`.
        The default is :class:`gpytorch.likelihoods.FixedNoiseGaussianLikelihood`.
    :param marginal_log_likelihood_class: An uninstantiated subclass of of an MLL from
        :mod:`gpytorch.mlls`. The default is :class:`gpytorch.mlls.ExactMarginalLogLikelihood`.
    :param optimiser_class: An uninstantiated :class:`torch.optim.Optimizer` class used for
        gradient-based learning of hyperparameters. The default is :class:`torch.optim.Adam`.
    :param smart_optimiser_class: An uninstantiated
        :class:`~vanguard.optimise.optimiser.SmartOptimiser` class used to wrap the
        ``optimiser_class`` and enable early stopping.
    :param rng: Generator instance used to generate random numbers.
    :param kwargs: For a complete list, see :class:`~vanguard.base.gpcontroller.GPController`.
    """

    def __init__(
        self,
        train_x: numpy.typing.NDArray[np.floating],
        train_y: numpy.typing.NDArray[np.floating],
        kernel_class: Type[gpytorch.kernels.Kernel],
        y_std: Union[numpy.typing.NDArray[np.floating], float],
        mean_class: Type[gpytorch.means.Mean] = ConstantMean,
        likelihood_class: Type[gpytorch.likelihoods.Likelihood] = FixedNoiseGaussianLikelihood,
        marginal_log_likelihood_class: Type[gpytorch.mlls.MarginalLogLikelihood] = ExactMarginalLogLikelihood,
        optimiser_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        smart_optimiser_class: Type[SmartOptimiser] = GreedySmartOptimiser,
        rng: Optional[np.random.Generator] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise self.
        """
        super().__init__(
            train_x=train_x,
            train_y=train_y,
            kernel_class=kernel_class,
            mean_class=mean_class,
            y_std=y_std,
            likelihood_class=likelihood_class,
            marginal_log_likelihood_class=marginal_log_likelihood_class,
            optimiser_class=optimiser_class,
            smart_optimiser_class=smart_optimiser_class,
            rng=utils.optional_random_generator(rng),
            **kwargs,
        )
