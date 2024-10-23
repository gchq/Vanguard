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
There are several pre-defined warp functions implementing some common maps.
"""

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import override

from vanguard.warps.basefunction import WarpFunction
from vanguard.warps.intermediate import require_controller_input


class AffineWarpFunction(WarpFunction):
    r"""
    A warp of form :math:`y \mapsto ay + b`.
    """

    def __init__(self, a: Union[float, int] = 1, b: Union[float, int] = 0) -> None:
        """
        Initialise self.

        :param a: The scale of the affine transformation.
        :param b: The shift of the affine transformation.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.as_tensor([[float(a)]]))
        self.bias = torch.nn.Parameter(torch.as_tensor([[float(b)]]))

    @property
    def a(self) -> torch.nn.Parameter:
        """Return the weight."""
        return self.weight

    @property
    def b(self) -> torch.nn.Parameter:
        """Return the bias."""
        return self.bias

    @override
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.a + self.b

    @override
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.div(x - self.b, self.a)

    @override
    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(y) * self.a


@require_controller_input("controller_inputs")
class PositiveAffineWarpFunction(AffineWarpFunction):
    r"""
    A warp of form :math:`y \mapsto ay + b`, where :math:`ay + b > 0`.

    .. note::
        This warp function needs to be activated before use.
        See :mod:`vanguard.warps.intermediate`.
    """

    def __init__(self, a: Union[float, int] = 1, b: Union[float, int] = 0) -> None:
        """
        Initialise self.

        :param a: The prior for the weight of the function.
        :param b: The prior for the bias of the function.
        """
        train_y = self.controller_inputs["train_y"]
        lambda_1, lambda_2 = self._get_constraint_slopes(train_y)
        beta_squared = (a * lambda_1 + b) / (lambda_2 - lambda_1)
        if beta_squared < 0:
            raise ValueError(
                "The supplied a and b values violate the constraints defined by the specified values of"
                f"lambda_1 and lambda_2, since a*lambda_1 + b < 0, i.e. {a}*{lambda_1} + {b} < 0."
            )

        beta = np.sqrt(beta_squared)
        alpha = np.sqrt(a + beta**2)

        super().__init__(alpha, beta)

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    @property
    def a(self) -> torch.Tensor:
        """Return the weight."""
        return self.weight**2 - self.bias**2

    @property
    def b(self) -> torch.Tensor:
        """Return the bias."""
        return -(self.weight**2 * self.lambda_1 - self.bias**2 * self.lambda_2)

    @staticmethod
    def _get_constraint_slopes(y_values: Union[Tensor, NDArray[np.floating]]) -> Tuple[float, float]:
        """
        Return the two constraint slopes needed for the y_values.

        :param y_values: A set of values for which :math:`ay + b` must ultimately hold.
        :returns: The two values needed to establish the same bounds on :math:`a` and :math:`b`.
        """
        y_values = torch.as_tensor(y_values)
        try:
            negative_contribution = y_values.min().item()
            non_negative_contribution = y_values.max().item()
        except RuntimeError:
            if y_values.numel() == 0:
                raise ValueError("Cannot process empty iterable.") from None
            else:
                raise
        else:
            return negative_contribution, non_negative_contribution


class BoxCoxWarpFunction(WarpFunction):
    r"""
    The Box-Cox warp as in :cite:`Rios19`.

    The transformation is given by:

    .. math::
        y\mapsto\frac{sgn(y)|y|^\lambda - 1}{\lambda}, \lambda\in\mathbb{R}_0^+.
    """

    def __init__(self, lambda_: Union[int, float] = 0) -> None:
        """
        Initialise self.

        :param lambda_: The parameter for the transformation.
        """
        super().__init__()
        self.lambda_ = lambda_

    @override
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.lambda_ == 0:
            return torch.log(y)
        else:
            return (torch.sign(y) * torch.abs(y) ** self.lambda_ - 1) / self.lambda_

    @override
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.lambda_ == 0:
            return torch.exp(x)
        else:
            return torch.sign(self.lambda_ * x + 1) * torch.abs(self.lambda_ * x + 1) ** (1 / self.lambda_)

    @override
    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        if self.lambda_ == 0:
            return 1 / y
        else:
            return torch.abs(y) ** (self.lambda_ - 1)


class SinhWarpFunction(WarpFunction):
    r"""
    A map of the form :math:`y\mapsto\sinh(y)`.
    """

    @override
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sinh(y)

    @override
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(x)

    @override
    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        return torch.cosh(y)


class ArcSinhWarpFunction(WarpFunction):
    r"""
    A map of the form :math:`y\mapsto\sinh^{-1}(y)`.
    """

    @override
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.asinh(y)

    @override
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinh(x)

    @override
    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        return 1 / torch.sqrt(y**2 + 1)


class LogitWarpFunction(WarpFunction):
    r"""
    A map of the form :math:`y\mapsto\log\frac{y}{1-y}`.
    """

    @override
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.logit(y)

    @override
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    @override
    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        return (1 - 2 * y) / (y * (1 - y))


class SoftPlusWarpFunction(WarpFunction):
    r"""
    A map of the form :math:`y\mapsto\log(e^y - 1)`.
    """

    @override
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.exp(y) - 1)

    @override
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.exp(x) + 1)

    @override
    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y)


AFFINE_LOG_WARP_FUNCTION: WarpFunction = BoxCoxWarpFunction(lambda_=0) @ AffineWarpFunction()
SAL_WARP_FUNCTION: WarpFunction = (
    AffineWarpFunction() @ SinhWarpFunction() @ AffineWarpFunction() @ ArcSinhWarpFunction()
)
