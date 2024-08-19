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

"""Tests for vanguard.variational.decorator."""

import pytest
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import MeanFieldVariationalDistribution, VariationalStrategy

from tests.cases import get_default_rng
from vanguard.datasets import Dataset
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference


@pytest.fixture(scope="module", name="dataset")
def get_dataset() -> Dataset:
    """A dataset to use for testing."""
    return SyntheticDataset(rng=get_default_rng(), n_test_points=10, n_train_points=20)


class TestWrapping:
    def test_variational_strategy_is_wrapped(self, dataset: Dataset):
        """Test that when a variational strategy class is passed, it is used."""

        class MyVariationalStrategy(VariationalStrategy):
            """Test class identical to `VariationalStrategy` in all but name."""

        @VariationalInference(variational_strategy_class=MyVariationalStrategy)
        class Controller(GaussianGPController):
            """GP controller for testing."""

        controller = Controller(
            train_x=dataset.train_x,
            train_y=dataset.train_y,
            y_std=dataset.train_y_std,
            kernel_class=ScaledRBFKernel,
            marginal_log_likelihood_class=VariationalELBO,
            rng=get_default_rng(),
        )

        # pylint: disable-next=protected-access
        assert isinstance(controller._gp.variational_strategy, MyVariationalStrategy)

    def test_variational_distribution_is_wrapped(self, dataset: Dataset):
        """Test that when a variational distribution class is passed, it is used."""

        class MyMeanFieldVariationalDistribution(MeanFieldVariationalDistribution):
            """Test class identical to `MeanFieldVariationalDistribution` in all but name."""

        @VariationalInference(variational_distribution_class=MyMeanFieldVariationalDistribution)
        class Controller(GaussianGPController):
            """GP controller for testing."""

        controller = Controller(
            train_x=dataset.train_x,
            train_y=dataset.train_y,
            y_std=dataset.train_y_std,
            kernel_class=ScaledRBFKernel,
            marginal_log_likelihood_class=VariationalELBO,
            rng=get_default_rng(),
        )

        assert isinstance(
            # pylint: disable-next=protected-access
            controller._gp.variational_strategy._variational_distribution,
            MyMeanFieldVariationalDistribution,
        )

    def test_variational_strategy_error_wrapping(self, dataset: Dataset):
        """Test that if the given strategy throws a `RuntimeError`, it is wrapped in an error with a nicer message."""

        class ErrorStrategy(VariationalStrategy):
            def __call__(self, *args, **kwargs):
                raise RuntimeError("test error")

        @VariationalInference(variational_strategy_class=ErrorStrategy)
        class Controller(GaussianGPController):
            """GP controller for testing."""

        controller = Controller(
            train_x=dataset.train_x,
            train_y=dataset.train_y,
            y_std=dataset.train_y_std,
            kernel_class=ScaledRBFKernel,
            marginal_log_likelihood_class=VariationalELBO,
            rng=get_default_rng(),
        )

        with pytest.raises(RuntimeError, match="may not be the correct choice for a variational strategy"):
            controller.fit()
