"""
Contains a slight adjustment to the standard multitask kernel.
"""

from typing import Any

from gpytorch.kernels import MultitaskKernel
from linear_operator import to_linear_operator
from linear_operator.operators import KroneckerProductLinearOperator
from torch import Tensor


class BatchCompatibleMultitaskKernel(MultitaskKernel):
    """
    A multitask kernel compatible with input uncertainty and hierarchical.
    """

    def forward(
        self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params: Any
    ) -> Tensor:
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")

        covar_i = self.task_covar_module.covar_matrix
        *leading_batch_dimensions, _, _ = x1.shape
        for _ in range(len(leading_batch_dimensions) - 1):
            covar_i = covar_i.unsqueeze(dim=0)
        covar_x = to_linear_operator(self.data_covar_module.forward(x1, x2, **params))
        res = KroneckerProductLinearOperator(covar_x, covar_i)
        return res.diagonal() if diag else res
