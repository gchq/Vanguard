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
Vanguard includes :class:`gpytorch.kernels.Kernel` subclasses which are recommended for use in controllers.
"""

from typing import Optional, Tuple, Union

import torch
from gpytorch import constraints, kernels


class ScaledRBFKernel(kernels.ScaleKernel):
    """
    The recommended starting place for a kernel.
    """

    def __init__(
        self, batch_shape: Union[Tuple[int], torch.Size] = torch.Size(), ard_num_dims: Optional[int] = None
    ) -> None:
        """
        Initialise self.

        :param batch_shape: The batch shape. Defaults to no batching.
        :param ard_num_dims: Set this if you want a separate lengthscale for each input dimension. Defaults to none.
        """
        super().__init__(kernels.RBFKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape), batch_shape=batch_shape)


class PeriodicRBFKernel(kernels.ScaleKernel):
    """
    An RBF kernel with a periodic element.
    """

    def __init__(self) -> None:
        """Initialise self."""
        super().__init__(kernels.RBFKernel() + kernels.ScaleKernel(kernels.RBFKernel() * kernels.PeriodicKernel()))


class TimeSeriesKernel(kernels.AdditiveKernel):
    """
    A kernel suited to time series.
    """

    def __init__(self, time_dimension: int = 0) -> None:
        """
        Initialise self.

        :param time_dimension: The dimension in the data that corresponds to time.
        """
        scaled_rbf_t = kernels.ScaleKernel(kernels.RBFKernel(active_dims=[time_dimension]))
        scaled_periodic_rbf = kernels.ScaleKernel(
            kernels.PeriodicKernel(active_dims=[time_dimension]) * kernels.RBFKernel(active_dims=[time_dimension])
        )
        scaled_constrained_rbf = kernels.ScaleKernel(
            kernels.RBFKernel(active_dims=[time_dimension]), lengthscale_constraint=constraints.Interval(1, 14)
        )
        scaled_linear_t = kernels.ScaleKernel(kernels.LinearKernel(active_dims=[time_dimension]))
        kernel_t = scaled_rbf_t + scaled_periodic_rbf + scaled_constrained_rbf

        super().__init__(scaled_linear_t, kernel_t)
