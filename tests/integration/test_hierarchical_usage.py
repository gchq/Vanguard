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
Basic end to end functionality test for hierarchical code in Vanguard.
"""

from typing import Tuple, Union

import numpy as np
import pytest
import torch
from gpytorch.kernels import RBFKernel
from numpy._typing import NDArray
from pytest import FixtureRequest
from torch import Tensor

from tests.cases import get_default_rng
from vanguard.hierarchical import (
    BayesianHyperparameters,
    LaplaceHierarchicalHyperparameters,
    VariationalHierarchicalHyperparameters,
)
from vanguard.hierarchical.base import BaseHierarchicalHyperparameters
from vanguard.vanilla import GaussianGPController

TrainTestData = Union[Tuple[NDArray, NDArray, NDArray, NDArray], Tuple[Tensor, Tensor, Tensor, Tensor]]


class TestHierarchicalUsage:
    """
    A subclass of TestCase designed to check end-to-end usage of hierarchical code.
    """

    num_train_points = 100
    num_test_points = 100
    n_sgd_iters = 100
    small_noise = 0.05
    num_mc_samples = 50

    @pytest.fixture(scope="class", params=["ndarray", "tensor"])
    def train_test_data(self, request: FixtureRequest) -> TrainTestData:
        rng = get_default_rng()

        # Define data for the tests
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.squeeze(x * np.sin(x))

        # Split data into training and testing
        train_indices = rng.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        x_train = x[train_indices]
        y_train = y[train_indices]
        x_test = x[test_indices]
        y_test = y[test_indices]

        if request.param == "ndarray":
            assert isinstance(x_train, np.ndarray)
            assert isinstance(x_test, np.ndarray)
            assert isinstance(y_train, np.ndarray)
            assert isinstance(x_test, np.ndarray)
        else:
            assert request.param == "tensor"
            x_train = torch.as_tensor(x_train)
            y_train = torch.as_tensor(y_train)
            x_test = torch.as_tensor(x_test)
            y_test = torch.as_tensor(y_test)

        return x_train, y_train, x_test, y_test

    @pytest.mark.parametrize(
        "hierarchical_decorator",
        [
            pytest.param(LaplaceHierarchicalHyperparameters(num_mc_samples=num_mc_samples), id="laplace"),
            pytest.param(VariationalHierarchicalHyperparameters(num_mc_samples=num_mc_samples), id="variational"),
        ],
    )
    def test_hierarchical(
        self, train_test_data: TrainTestData, hierarchical_decorator: BaseHierarchicalHyperparameters
    ) -> None:
        """
        Verify Vanguard usage on a simple, single variable regression problem when using hierarchical hyperparameters.

        We test with the LaplaceHierarchicalHyperparameters and VariationalHierarchicalHyperparameters decorators
        separately.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        GP can be fit to this data.
        """
        train_x, train_y, test_x, _ = train_test_data

        # Create a hierarchical controller and a kernel that has Bayesian hyperparameters to estimate
        @hierarchical_decorator
        class HierarchicalController(GaussianGPController):
            pass

        @BayesianHyperparameters()
        class BayesianRBFKernel(RBFKernel):
            pass

        # Define the controller object
        gp = HierarchicalController(
            train_x=train_x,
            train_y=train_y,
            kernel_class=BayesianRBFKernel,
            y_std=self.small_noise,
            rng=get_default_rng(),
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        prediction_medians, prediction_ci_lower, prediction_ci_upper = gp.posterior_over_point(
            test_x
        ).confidence_interval()

        # Sense check the outputs. Note that we do not check confidence interval quality here,
        # just that they can be created, due to highly varying quality of the resulting intervals,
        assert torch.all(prediction_medians <= prediction_ci_upper)
        assert torch.all(prediction_medians >= prediction_ci_lower)
