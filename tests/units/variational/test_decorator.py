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
            """Test class identical to VariationalStrategy in all but name."""

        @VariationalInference(variational_strategy_class=MyVariationalStrategy)
        class Controller(GaussianGPController):
            """GP controller for testing."""

        controller = Controller(
            train_x=dataset.train_x,
            train_y=dataset.train_y,
            y_std=dataset.train_y_std,
            kernel_class=ScaledRBFKernel,
            marginal_log_likelihood_class=VariationalELBO,
        )

        # pylint: disable-next=protected-access
        assert isinstance(controller._gp.variational_strategy, MyVariationalStrategy)

    def test_variational_distribution_is_wrapped(self, dataset: Dataset):
        """Test that when a variational distribution class is passed, it is used."""

        class MyMeanFieldVariationalDistribution(MeanFieldVariationalDistribution):
            """Test class identical to MeanFieldVariationalDistribution in all but name."""

        @VariationalInference(variational_distribution_class=MyMeanFieldVariationalDistribution)
        class Controller(GaussianGPController):
            """GP controller for testing."""

        controller = Controller(
            train_x=dataset.train_x,
            train_y=dataset.train_y,
            y_std=dataset.train_y_std,
            kernel_class=ScaledRBFKernel,
            marginal_log_likelihood_class=VariationalELBO,
        )

        assert isinstance(
            # pylint: disable-next=protected-access
            controller._gp.variational_strategy._variational_distribution,
            MyMeanFieldVariationalDistribution,
        )
