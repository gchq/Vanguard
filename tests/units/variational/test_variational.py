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
Test the behaviour of the VariationalInference decorator.
"""

import unittest

from gpytorch.mlls import ExactMarginalLogLikelihood as InappropriateMarginalLogLikelihood

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference


@VariationalInference()
class VariationalGPController(GaussianGPController):
    pass


class BasicTests(unittest.TestCase):
    """
    Basic tests for the decorator.
    """

    def test_bad_marginal_log_likelihood(self) -> None:
        """
        Test that when an inappropriate MLL class is passed, the resulting `TypeError` is converted to a `ValueError`.
        """
        rng = get_default_rng()
        dataset = SyntheticDataset(rng=rng)
        with self.assertRaises(TypeError) as ctx:
            VariationalGPController(
                dataset.train_x,
                dataset.train_y,
                ScaledRBFKernel,
                dataset.train_y_std,
                marginal_log_likelihood_class=InappropriateMarginalLogLikelihood,
                rng=rng,
            )

        assert str(ctx.exception) == (
            "The class passed to `marginal_log_likelihood_class` must take a "
            "`num_data: int` argument, since we run variational inference with SGD."
        )

    def test_other_type_error_unaffected(self):
        """Test that any other `TypeError` is raised as-is and is not converted to a `ValueError`."""
        rng = get_default_rng()
        dataset = SyntheticDataset(rng=rng)
        with self.assertRaises(TypeError):
            VariationalGPController(
                dataset.train_x,
                dataset.train_y,
                ScaledRBFKernel,
                dataset.train_y_std,
                marginal_log_likelihood_class="incorrect type",
                rng=rng,
            )
