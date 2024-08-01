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
