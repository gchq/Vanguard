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
Basic end to end functionality test for distributed decorators in Vanguard.
"""

from typing import Tuple, Type, Union

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from numpy.typing import NDArray
from sklearn.metrics import f1_score
from torch import Tensor

from tests.cases import get_default_rng
from tests.integration.util import train_test_split_convert
from vanguard.classification import BinaryClassification
from vanguard.distribute import Distributed
from vanguard.distribute.aggregators import (
    BaseAggregator,
    BCMAggregator,
    EKPOEAggregator,
    GPOEAggregator,
    GRBCMAggregator,
    POEAggregator,
    RBCMAggregator,
    XBCMAggregator,
    XGRBCMAggregator,
)
from vanguard.distribute.partitioners import (
    BasePartitioner,
    KMeansPartitioner,
    KMedoidsPartitioner,
    MiniBatchKMeansPartitioner,
    RandomPartitioner,
)
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference

TrainTestData = Union[Tuple[NDArray, NDArray, NDArray, NDArray], Tuple[Tensor, Tensor, Tensor, Tensor]]


class TestDistributeUsage:
    """
    A subclass of TestCase designed to check end-to-end usage of distributed code.
    """

    num_train_points = 100
    num_test_points = 100
    n_sgd_iters = 50
    # How high of an F1 score do we need to consider the test a success (and a fit
    # successful?)
    required_f1_score = 0.9

    @pytest.fixture(scope="class", params=["ndarray", "tensor"])
    def binary_classification_data(self, request: FixtureRequest) -> TrainTestData:
        """Generate binary classification data for testing."""
        rng = get_default_rng()

        # Define some data for the test
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.zeros_like(x)
        for index, x_val in enumerate(x):
            # Set some non-trivial classification target
            if 0.25 < x_val < 0.5:
                y[index, 0] = 1
            if x_val > 0.8:
                y[index, 0] = 1

        x_train, x_test, y_train, y_test = train_test_split_convert(
            x, y, n_test_points=self.num_test_points, array_type=request.param, rng=rng
        )

        return x_train, y_train, x_test, y_test

    @pytest.mark.parametrize(
        "aggregator",
        [
            EKPOEAggregator,
            GPOEAggregator,
            BCMAggregator,
            RBCMAggregator,
            XBCMAggregator,
            GRBCMAggregator,
            XGRBCMAggregator,
            POEAggregator,
        ],
    )
    @pytest.mark.parametrize(
        "partitioner",
        [
            RandomPartitioner,
            KMeansPartitioner,
            MiniBatchKMeansPartitioner,
            KMedoidsPartitioner,
        ],
    )
    def test_distributed_gp_vary_aggregator_and_partitioner(
        self,
        binary_classification_data: TrainTestData,
        aggregator: Type[BaseAggregator],
        partitioner: Type[BasePartitioner],
    ) -> None:
        """
        Verify Vanguard usage on a simple, single variable distributed binary classification problem
        using the various aggregators and partition methods.

        We generate a single feature `x` and a binary target `y`, and verify that a
        GP can be fit to this data.
        """
        train_x, train_y, test_x, test_y = binary_classification_data

        # We have a binary classification problem, so we apply the BinaryClassification
        # decorator and will need to use VariationalInference to perform inference on
        # data. We try each aggregation method to ensure they are all functional.

        @Distributed(n_experts=3, aggregator_class=aggregator, partitioner_class=partitioner, rng=get_default_rng())
        @BinaryClassification()
        @VariationalInference()
        class BinaryClassifier(GaussianGPController):
            pass

        if partitioner is KMedoidsPartitioner:
            # KMedoids requires a kernel to be passed to the partitioner
            partitioner_kwargs = {"kernel": ScaledRBFKernel()}
        else:
            partitioner_kwargs = {}

        # Define the controller object
        gp = BinaryClassifier(
            train_x=train_x,
            train_y=train_y,
            kernel_class=ScaledRBFKernel,
            y_std=0.0,
            likelihood_class=BernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            partitioner_kwargs=partitioner_kwargs,
            rng=get_default_rng(),
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        predictions_train, _ = gp.classify_points(train_x)
        predictions_test, _ = gp.classify_points(test_x)

        # Sense check outputs
        assert f1_score(predictions_train, train_y) >= self.required_f1_score
        assert f1_score(predictions_test, test_y) >= self.required_f1_score
