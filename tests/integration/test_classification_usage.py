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
Basic end to end functionality test for classification problems in Vanguard.
"""

from typing import Literal, Union

import gpytorch.means
import numpy as np
import pytest
from gpytorch.likelihoods import BernoulliLikelihood, DirichletClassificationLikelihood
from gpytorch.mlls import VariationalELBO
from numpy.typing import NDArray
from sklearn.metrics import f1_score
from torch import Tensor

from tests.cases import get_default_rng
from tests.integration.util import train_test_split_convert
from vanguard.classification import BinaryClassification, DirichletMulticlassClassification
from vanguard.classification.kernel import DirichletKernelMulticlassClassification
from vanguard.classification.likelihoods import DirichletKernelClassifierLikelihood, GenericExactMarginalLogLikelihood
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference

TrainTestData = Union[tuple[NDArray, NDArray, NDArray, NDArray], tuple[Tensor, Tensor, Tensor, Tensor]]


class TestClassification:
    """
    A subclass of TestCase designed to check end-to-end usage of classification code.
    """

    num_train_points = 100
    num_test_points = 100
    n_sgd_iters = 50
    # How high of an F1 score do we need to consider the test a success (and a fit
    # successful?)
    required_f1_score = 0.9

    @pytest.fixture(scope="class", params=["ndarray", "tensor"])
    def train_test_data_multiclass(self, request: pytest.FixtureRequest) -> TrainTestData:
        """Create multiclass train/test data."""
        return self.make_data(request.param, "multiclass")

    @pytest.fixture(scope="class", params=["ndarray", "tensor"])
    def train_test_data_binary(self, request: pytest.FixtureRequest) -> TrainTestData:
        """Create binary train/test data."""
        return self.make_data(request.param, "binary")

    def make_data(
        self, array_type: Literal["ndarray", "tensor"], classification_type: Literal["binary", "multiclass"]
    ) -> TrainTestData:
        """
        Create classification data for testing.

        :param array_type: either "ndarray" or "tensor", depending on the desired output type
        :param classification_type: either "binary" or "multiclass", depending on the desired number of output classes
        :return: Tuple (x_train, y_train, x_test, y_test)
        """
        rng = get_default_rng()

        # Define some data for the test
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.zeros_like(x, dtype=np.integer)

        # Set some non-trivial classification target, with either 2 or 3 classes depending on `classification_type`
        third_class_value = 1 if classification_type == "binary" else 2
        for index, x_val in enumerate(x):
            if 0.25 < x_val < 0.5:
                y[index, 0] = 1
            if x_val > 0.8:
                y[index, 0] = third_class_value

        x_train, x_test, y_train, y_test = train_test_split_convert(
            x, y, n_test_points=self.num_test_points, array_type=array_type, rng=rng
        )

        return x_train, y_train, x_test, y_test

    def test_gp_binary_classification(self, train_test_data_binary: TrainTestData) -> None:
        """
        Verify Vanguard usage on a simple, single variable binary classification problem.

        We generate a single feature `x` and a binary target `y`, and verify that a
        GP can be fit to this data.
        """
        # Unpack train/test data
        train_x, train_y, test_x, test_y = train_test_data_binary

        # We have a binary classification problem, so we apply the BinaryClassification
        # decorator and will need to use VariationalInference to perform inference on
        # data
        @BinaryClassification()
        @VariationalInference()
        class BinaryClassifier(GaussianGPController):
            pass

        # Define the controller object
        gp = BinaryClassifier(
            train_x=train_x,
            train_y=train_y,
            kernel_class=ScaledRBFKernel,
            y_std=0.0,
            likelihood_class=BernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
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

    def test_gp_categorical_classification_exact(self, train_test_data_multiclass: TrainTestData) -> None:
        """
        Verify Vanguard usage on a simple, single variable categorical classification problem
        using exact inference.

        We generate a single feature `x` and a categorical target `y`, and verify that a
        GP can be fit to this data.
        """
        # Unpack train/test data
        train_x, train_y, test_x, test_y = train_test_data_multiclass

        # We have a multi-class classification problem, so we apply the DirichletMulticlassClassification
        # decorator to perform exact inference
        @DirichletMulticlassClassification(num_classes=3)
        class CategoricalClassifier(GaussianGPController):
            pass

        # Define the controller object
        gp = CategoricalClassifier(
            train_x=train_x,
            train_y=train_y[:, 0],
            kernel_class=ScaledRBFKernel,
            y_std=0.0,
            kernel_kwargs={"batch_shape": (3,)},
            likelihood_class=DirichletClassificationLikelihood,
            rng=get_default_rng(),
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        predictions_train, _ = gp.classify_points(train_x)
        predictions_test, _ = gp.classify_points(test_x)

        # Sense check outputs
        assert f1_score(predictions_train, train_y, average="micro") >= self.required_f1_score
        assert f1_score(predictions_test, test_y, average="micro") >= self.required_f1_score

    def test_gp_categorical_classification_dirichlet_kernel(self, train_test_data_multiclass: TrainTestData) -> None:
        """
        Verify Vanguard usage on a simple, single variable categorical classification problem
        using DirichletKernelMulticlassClassification.

        We generate a single feature `x` and a categorical target `y`, and verify that a
        GP can be fit to this data.
        """
        # Unpack train/test data
        train_x, train_y, test_x, test_y = train_test_data_multiclass

        # We have a multi-class classification problem, so we apply the DirichletKernelMulticlassClassification
        # decorator
        @DirichletKernelMulticlassClassification(num_classes=3, ignore_methods=("__init__",))
        class CategoricalClassifier(GaussianGPController):
            pass

        # Define the controller object
        gp = CategoricalClassifier(
            train_x=train_x,
            train_y=train_y[:, 0],
            kernel_class=ScaledRBFKernel,
            y_std=0.0,
            likelihood_class=DirichletKernelClassifierLikelihood,
            marginal_log_likelihood_class=GenericExactMarginalLogLikelihood,
            rng=get_default_rng(),
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        predictions_train, _ = gp.classify_points(train_x)
        predictions_test, _ = gp.classify_points(test_x)

        # Sense check outputs
        assert f1_score(predictions_train, train_y, average="micro") >= self.required_f1_score
        assert f1_score(predictions_test, test_y, average="micro") >= self.required_f1_score

    def test_multitask_dirichlet_classification_notebook_example(self) -> None:
        """
        Explicitly test the multitask Dirichlet classification example using Gaussian data.

        Here we recreate a minimal example based on the content of the notebook  ``multiclass_dirichlet_classification``
        to ensure it is tested on all supported python versions.
        """
        # Recreate the notebook example with specific data
        num_classes = 4
        dataset = MulticlassGaussianClassificationDataset(
            num_train_points=100,
            num_test_points=500,
            num_classes=num_classes,
            covariance_scale=1.0,
            rng=get_default_rng(),
        )

        # Test the first controller can be created and fit without error
        @DirichletMulticlassClassification(num_classes=num_classes, ignore_methods=("__init__",))
        class MulticlassGaussianClassifierWithoutKernel(GaussianGPController):
            pass

        gp = MulticlassGaussianClassifierWithoutKernel(
            dataset.train_x,
            dataset.train_y,
            ScaledRBFKernel,
            y_std=0.0,
            mean_class=gpytorch.means.ZeroMean,
            likelihood_class=DirichletClassificationLikelihood,
            mean_kwargs={"batch_shape": (num_classes,)},
            kernel_kwargs={"batch_shape": (num_classes,)},
            likelihood_kwargs={"alpha_epsilon": 0.3, "learn_additional_noise": True},
            optim_kwargs={"lr": 0.05},
            rng=get_default_rng(),
        )
        gp.fit(1)

    def test_multitask_dirichlet_kernel_classification_notebook_example(self):
        """
        Explicitly test the multitask Dirichlet kernel classification example using Gaussian data.

        Here we recreate a minimal example based on the content of the notebook  ``multiclass_dirichlet_classification``
        to ensure it is tested on all supported python versions.
        """
        # Recreate the notebook example with specific data
        num_classes = 4
        dataset = MulticlassGaussianClassificationDataset(
            num_train_points=100,
            num_test_points=500,
            num_classes=num_classes,
            covariance_scale=1.0,
            rng=get_default_rng(),
        )

        # Test the second controller can be created and fit without error
        @DirichletKernelMulticlassClassification(num_classes=num_classes, ignore_methods=("__init__",))
        class MulticlassGaussianClassifierWithKernel(GaussianGPController):
            pass

        gp = MulticlassGaussianClassifierWithKernel(
            dataset.train_x,
            dataset.train_y,
            kernel_class=ScaledRBFKernel,
            y_std=0.0,
            mean_class=gpytorch.means.ZeroMean,
            likelihood_class=DirichletKernelClassifierLikelihood,
            likelihood_kwargs={"learn_alpha": False, "alpha": 5.0},
            marginal_log_likelihood_class=GenericExactMarginalLogLikelihood,
            optim_kwargs={"lr": 0.1, "early_stop_patience": 5},
            rng=get_default_rng(),
        )
        gp.fit(1)
