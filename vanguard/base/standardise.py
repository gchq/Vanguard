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
Contains a class decorator to apply input standard scaling to means and kernels.
"""

from typing import Any, Optional, Type, Union

import numpy.typing
import torch
from typing_extensions import Self

from vanguard.decoratorutils import wraps_class


class StandardiseXModule:
    """
    A simple decorator to standard scale the inputs to a mean or kernel module before applying the mean or kernel.
    """

    def __init__(
        self,
        mean: Union[torch.FloatTensor, numpy.typing.NDArray[float], float],
        scale: Union[torch.FloatTensor, numpy.typing.NDArray[float], float],
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> None:
        """
        Initialise self.

        :param mean: The mean (i.e. additive shift) of the standard scaling.
                Can be an array in the case of multiple features.
        :param scale: The scale (i.e. standard deviation) of the standard scaling.
                Can be an array in the case of multiple features.
        :param device: The device on which the mean and scale parameters should live.
        :param dtype: Datatype to specify when creating torch tensors.
        """
        self.mean = torch.as_tensor(mean, device=device, dtype=dtype)
        self.scale = torch.as_tensor(scale, device=device, dtype=dtype)

    def apply(
        self,
        module_class: Type[torch.nn.Module],
    ) -> Type[torch.nn.Module]:
        """
        Modify the module's forward method to include standard scaling.

        :param module_class: The mean or kernel class to standard scale.
        :returns: The modified module class.
        """
        mean, scale = self.mean, self.scale

        @wraps_class(module_class)
        class ScaledModule(module_class):
            """An inner class which scales the forward method."""

            def forward(self, *args: Any, **kwargs: Any):
                """Scale the inputs before being passed."""
                scaled_args = ((arg - mean) / scale for arg in args)
                return super().forward(*scaled_args, **kwargs)

        return ScaledModule

    @classmethod
    def from_data(
        cls,
        x: torch.Tensor,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> Self:
        """
        Create an instance of self with the mean and scale of the standard scaling obtained from the given data.

        :param x: (n_sample, n_features) The input data on which to learn to mean and scale.
        :param device: Where the mean and scale will reside.
        :param dtype: Datatype to specify when creating torch tensors.
        """
        mean, scale = x.mean(dim=0), x.std(dim=0)
        return cls(mean, scale, device, dtype)
