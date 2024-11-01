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
Contains a torch distribution implementing a warped Gaussian.
"""

from typing import Any, Union

import numpy as np
import numpy.typing
import torch
from torch.distributions import Normal
from typing_extensions import Self, override

from vanguard.base.basecontroller import BaseGPController
from vanguard.warps.basefunction import WarpFunction


class WarpedGaussian(Normal):
    r"""
    A warped Gaussian distribution.

    .. math::
        X\sim \mathcal{WN}(\psi; \mu, \sigma) ~ \iff  \psi(X)\sim\mathcal{N}(\mu, \sigma).
    """

    def __init__(self, warp: WarpFunction, *args: Any, **kwargs: Any) -> None:
        """
        :param warp`: The warp to be used to define the distribution.
        """
        super().__init__(*args, **kwargs)
        self.warp = warp

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-probability of the values under the warped Gaussian distribution.

        :param value: Shape should be compatible with the distributions shape.
        :returns: The log probability of the values.
        """
        gaussian = super().log_prob(self.warp(value))
        jacobian = torch.log(self.warp.deriv(value).abs())
        return gaussian + jacobian

    def sample(self, *args: Any, **kwargs: Any):
        """
        Sample from the distribution.
        """
        gaussian_samples = super().sample(*args, **kwargs)
        return self.warp.inverse(gaussian_samples)

    @classmethod
    def from_data(
        cls,
        warp: WarpFunction,
        samples: Union[torch.Tensor, numpy.typing.NDArray[np.floating]],
        optimiser: type[torch.optim.Optimizer] = torch.optim.Adam,
        n_iterations: int = 100,
        lr: float = 0.001,
    ) -> Self:
        """
        Fit a warped Gaussian distribution to the given data using the supplied warp.

        The mean and variance will be
        optimised along with the free parameters of the warp.

        :param warp: The warp to use.
        :param samples: (n_samples, ...) The data to fit.
        :param optimiser: A subclass of :class:`torch.optim.Optimizer` used to tune the parameters.
        :param n_iterations: The number of optimisation iterations.
        :param lr: The learning rate for optimisation.
        :returns: A fit distribution.
        """
        t_samples = torch.as_tensor(samples, dtype=BaseGPController.get_default_tensor_dtype())
        optim = optimiser(params=[{"params": warp.parameters(), "lr": lr}])  # pyright: ignore [reportCallIssue]

        for i in range(n_iterations):
            loss = -cls._mle_log_prob_parametrised_with_warp_parameters(warp, t_samples)
            loss.backward(retain_graph=i < n_iterations - 1)
            optim.step()
        w_samples = warp(t_samples)
        loc = w_samples.mean(dim=0).detach()  # pyright: ignore [reportCallIssue]
        scale = w_samples.std(dim=0).detach() + 1e-4
        distribution = cls(warp, loc=loc, scale=scale)

        return distribution

    @staticmethod
    def _mle_log_prob_parametrised_with_warp_parameters(warp: WarpFunction, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of the data under the warped Gaussian.

        This is done using the optimal MLEs for the Gaussian mean and variance
        parameters, leaving only a function of the warp parameters.
        """
        w_data = warp(data)
        loc = w_data.mean(dim=0).detach()
        scale = w_data.std(dim=0).detach() + 1e-4
        gaussian_log_prob = (
            -((w_data - loc) ** 2) / (2 * scale**2) - torch.log(scale)  # pyright: ignore [reportOperatorIssue]
        ).sum()
        log_jacobian = torch.log(warp.deriv(data).abs()).sum()
        return gaussian_log_prob + log_jacobian

    @override
    def enumerate_support(self, expand: bool = True) -> torch.Tensor:
        raise NotImplementedError
