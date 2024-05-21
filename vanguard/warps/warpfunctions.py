"""
There are several pre-defined warp functions implementing some common maps.
"""

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional

from .basefunction import WarpFunction
from .intermediate import require_controller_input


class AffineWarpFunction(WarpFunction):
    r"""
    A warp of form :math:`y \mapsto ay + b`.
    """

    def __init__(self, a: Union[float, int] = 1, b: Union[float, int] = 0):
        """
        Initialise self.

        :param a: The scale of the affine transformation.
        :param b: The shift of the affine transformation.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor([[float(a)]]))
        self.bias = torch.nn.Parameter(torch.Tensor([[float(b)]]))

    @property
    def a(self) -> torch.nn.Parameter:
        """Return the weight."""
        return self.weight

    @property
    def b(self) -> torch.nn.Parameter:
        """Return the bias."""
        return self.bias

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.a + self.b

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.div(x - self.b, self.a)

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

    def __init__(self, a: Union[float, int] = 1, b: Union[float, int] = 0):
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
    def a(self) -> torch.nn.Parameter:
        """Return the weight."""
        return self.weight**2 - self.bias**2

    @property
    def b(self) -> torch.nn.Parameter:
        """Return the bias."""
        return -(self.weight**2 * self.lambda_1 - self.bias**2 * self.lambda_2)

    @staticmethod
    def _get_constraint_slopes(y_values: np.typing.NDArray[np.floating]) -> Tuple[np.floating, np.floating]:
        """
        Return the two constraint slopes needed for the y_values.

        :param y_values: A set of values for which :math:`ay + b` must ultimately hold.
        :returns: The two values needed to establish the same bounds on :math:`a` and :math:`b`.
        """
        try:
            negative_contribution = min(y_values)
            non_negative_contribution = max(y_values)
        except ValueError:
            raise ValueError("Cannot process empty iterable.") from None
        else:
            try:
                return negative_contribution[0], non_negative_contribution[0]
            except IndexError:
                return negative_contribution, non_negative_contribution


class BoxCoxWarpFunction(WarpFunction):
    r"""
    The Box-Cox warp as in :cite:`Rios19`.

    The transformation is given by:

    .. math::
        y\mapsto\frac{sgn(y)|y|^\lambda - 1}{\lambda}, \lambda\in\mathbb{R}_0^+.
    """

    def __init__(self, lambda_: float = 0):
        """
        Initialise self.

        :param lambda_: The parameter for the transformation.
        """
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.lambda_ == 0:
            return torch.log(y)
        else:
            return (torch.sign(y) * torch.abs(y) ** self.lambda_ - 1) / self.lambda_

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.lambda_ == 0:
            return torch.exp(x)
        else:
            return torch.sign(self.lambda_ * x + 1) * torch.abs(self.lambda_ * x + 1) ** (1 / self.lambda_)

    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        if self.lambda_ == 0:
            return 1 / y
        else:
            return torch.abs(y) ** (self.lambda_ - 1)


class SinhWarpFunction(WarpFunction):
    r"""
    A map of the form :math:`y\mapsto\sinh(y)`.
    """

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sinh(y)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(x)

    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        return torch.cosh(y)


class ArcSinhWarpFunction(WarpFunction):
    r"""
    A map of the form :math:`y\mapsto\sinh^{-1}(y)`.
    """

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.asinh(y)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinh(x)

    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        return 1 / torch.sqrt(y**2 + 1)


class LogitWarpFunction(WarpFunction):
    r"""
    A map of the form :math:`y\mapsto\log\frac{y}{1-y}`.
    """

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.logit(y)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        return (1 - 2 * y) / (y * (1 - y))


class SoftPlusWarpFunction(WarpFunction):
    r"""
    A map of the form :math:`y\mapsto\log(e^y - 1)`.
    """

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.exp(y) - 1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.exp(x) + 1)

    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y)


AFFINE_LOG_WARP_FUNCTION: WarpFunction = BoxCoxWarpFunction(lambda_=0) @ AffineWarpFunction()
SAL_WARP_FUNCTION: WarpFunction = (
    AffineWarpFunction() @ SinhWarpFunction() @ AffineWarpFunction() @ ArcSinhWarpFunction()
)
