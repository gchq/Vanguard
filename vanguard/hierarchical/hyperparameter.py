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
Contains the BayesianHyperparameter class.
"""

from typing import Optional

import torch
from gpytorch import constraints


class BayesianHyperparameter:
    """
    Represents a single Bayesian hyperparameter.
    """

    def __init__(
        self,
        raw_name: str,
        raw_shape: torch.Size,
        constraint: Optional[constraints.Interval],
        prior_mean: float,
        prior_variance: float,
    ) -> None:
        """
        Initialise self.

        :param raw_name: The raw name for the parameter.
        :param raw_shape: The shape of the raw parameter.
        :param constraint: The constraint (if any) placed on the parameter.
        :param prior_mean: The mean of the diagonal normal prior on the raw parameter.
        :param prior_variance: The variance of the diagonal normal prior on the raw parameter.
        """
        self.raw_name = raw_name
        self.raw_shape = raw_shape
        self.constraint = constraint
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

    def numel(self) -> int:
        """Return the number of elements in the parameter."""
        return self.raw_shape.numel()
