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

import unittest

import gpytorch.means
import numpy as np
from gpytorch.likelihoods import BernoulliLikelihood, DirichletClassificationLikelihood
from gpytorch.mlls import VariationalELBO
from sklearn.metrics import f1_score

from tests.cases import get_default_rng
from vanguard.classification import BinaryClassification, DirichletMulticlassClassification
from vanguard.classification.kernel import DirichletKernelMulticlassClassification
from vanguard.classification.likelihoods import DirichletKernelClassifierLikelihood, GenericExactMarginalLogLikelihood
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference


class VanguardTestCase(unittest.TestCase):
    """
    A subclass of TestCase designed to check end-to-end usage of classification code.
    """

    def setUp(self) -> None:
        """
        Define data shared across tests.
        """
        self.rng = get_default_rng()
        self.num_train_points = 100
        self.num_test_points = 100
        self.n_sgd_iters = 50
        # How high of an F1 score do we need to consider the test a success (and a fit
        # successful?)
        self.required_f1_score = 0.9

    def test_gp_binary_classification(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable binary classification problem.

        We generate a single feature `x` and a binary target `y`, and verify that a
        GP can be fit to this data.
        """
        # Define some data for the test
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.zeros_like(x)
        for index, x_val in enumerate(x):
            # Set some non-trivial classification target
            if 0.25 < x_val < 0.5:
                y[index, 0] = 1
            if x_val > 0.8:
                y[index, 0] = 1

        # Split data into training and testing
        train_indices = self.rng.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # We have a binary classification problem, so we apply the BinaryClassification
        # decorator and will need to use VariationalInference to perform inference on
        # data
        @BinaryClassification()
        @VariationalInference()
        class BinaryClassifier(GaussianGPController):
            pass

        # Define the controller object
        gp = BinaryClassifier(
            train_x=x[train_indices],
            train_y=y[train_indices],
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=BernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            rng=self.rng,
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        predictions_train, _ = gp.classify_points(x[train_indices])
        predictions_test, _ = gp.classify_points(x[test_indices])

        # Sense check outputs
        self.assertGreaterEqual(f1_score(predictions_train, y[train_indices]), self.required_f1_score)
        self.assertGreaterEqual(f1_score(predictions_test, y[test_indices]), self.required_f1_score)

    def test_gp_categorical_classification_exact(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable categorical classification problem
        using exact inference.

        We generate a single feature `x` and a categorical target `y`, and verify that a
        GP can be fit to this data.
        """
        # Define some data for the test
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.zeros_like(x, dtype=np.integer)
        for index, x_val in enumerate(x):
            # Set some non-trivial classification target with 3 classes (0, 1 and 2)
            if 0.25 < x_val < 0.5:
                y[index, 0] = 1
            if x_val > 0.8:
                y[index, 0] = 2

        # Split data into training and testing
        train_indices = self.rng.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # We have a multi-class classification problem, so we apply the DirichletMulticlassClassification
        # decorator to perform exact inference
        @DirichletMulticlassClassification(num_classes=3)
        class CategoricalClassifier(GaussianGPController):
            pass

        # Define the controller object
        gp = CategoricalClassifier(
            train_x=x[train_indices],
            train_y=y[train_indices, 0],
            kernel_class=ScaledRBFKernel,
            y_std=0,
            kernel_kwargs={"batch_shape": (3,)},
            likelihood_class=DirichletClassificationLikelihood,
            rng=self.rng,
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        predictions_train, _ = gp.classify_points(x[train_indices])
        predictions_test, _ = gp.classify_points(x[test_indices])

        # Sense check outputs
        self.assertGreaterEqual(f1_score(predictions_train, y[train_indices], average="micro"), self.required_f1_score)
        self.assertGreaterEqual(f1_score(predictions_test, y[test_indices], average="micro"), self.required_f1_score)

    def test_gp_categorical_classification_dirichlet_kernel(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable categorical classification problem
        using DirichletKernelMulticlassClassification.

        We generate a single feature `x` and a categorical target `y`, and verify that a
        GP can be fit to this data.
        """
        # Define some data for the test
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.zeros_like(x)
        for index, x_val in enumerate(x):
            # Set some non-trivial classification target with 3 classes (0, 1 and 2)
            if 0.25 < x_val < 0.5:
                y[index, 0] = 1
            if x_val > 0.8:
                y[index, 0] = 2

        # Split data into training and testing
        train_indices = self.rng.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # We have a multi-class classification problem, so we apply the DirichletKernelMulticlassClassification
        # decorator
        @DirichletKernelMulticlassClassification(num_classes=3, ignore_methods=("__init__",))
        class CategoricalClassifier(GaussianGPController):
            pass

        # Define the controller object
        gp = CategoricalClassifier(
            train_x=x[train_indices],
            train_y=y[train_indices, 0],
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=DirichletKernelClassifierLikelihood,
            marginal_log_likelihood_class=GenericExactMarginalLogLikelihood,
            rng=self.rng,
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        predictions_train, _ = gp.classify_points(x[train_indices])
        predictions_test, _ = gp.classify_points(x[test_indices])

        # Sense check outputs
        self.assertGreaterEqual(f1_score(predictions_train, y[train_indices], average="micro"), self.required_f1_score)
        self.assertGreaterEqual(f1_score(predictions_test, y[test_indices], average="micro"), self.required_f1_score)

    def test_multitask_dirichlet_classification_notebook_example(self) -> None:
        """
        Explicitly test the multitask dirichlet classification example using Gaussian data.

        Here we recreate a minimal example based on the content of the notebook  ``multiclass_dirichlet_classification''
        to ensure it is tested on all python versions rather than just a pinned package version and the latest.
        """
        # Recreate the notebook example with specific data
        num_classes = 4
        dataset = MulticlassGaussianClassificationDataset(
            num_train_points=100,
            num_test_points=500,
            num_classes=num_classes,
            covariance_scale=1,
            rng=self.rng,
        )

        @DirichletKernelMulticlassClassification(num_classes=num_classes, ignore_methods=("__init__",))
        class MulticlassGaussianClassifierWithKernel(GaussianGPController):
            pass

        @DirichletMulticlassClassification(num_classes=num_classes, ignore_methods=("__init__",))
        class MulticlassGaussianClassifierWithoutKernel(GaussianGPController):
            pass

        # Test the first controller can be created and fit
        gp = MulticlassGaussianClassifierWithoutKernel(
            dataset.train_x,
            dataset.train_y,
            ScaledRBFKernel,
            y_std=0,
            mean_class=gpytorch.means.ZeroMean,
            likelihood_class=DirichletClassificationLikelihood,
            mean_kwargs={"batch_shape": (num_classes,)},
            kernel_kwargs={"batch_shape": (num_classes,)},
            likelihood_kwargs={"alpha_epsilon": 0.3, "learn_additional_noise": True},
            optim_kwargs={"lr": 0.05},
            rng=self.rng,
        )
        gp.fit(1)

        # Test the second controller can be created and fit
        gp = MulticlassGaussianClassifierWithKernel(
            dataset.train_x,
            dataset.train_y,
            kernel_class=ScaledRBFKernel,
            y_std=0,
            mean_class=gpytorch.means.ZeroMean,
            likelihood_class=DirichletKernelClassifierLikelihood,
            likelihood_kwargs={"learn_alpha": False, "alpha": 5},
            marginal_log_likelihood_class=GenericExactMarginalLogLikelihood,
            optim_kwargs={"lr": 0.1, "early_stop_patience": 5},
            rng=self.rng,
        )
        gp.fit(1)

        # Test the third controller can be created and fit
        gp = MulticlassGaussianClassifierWithKernel(
            dataset.train_x,
            dataset.train_y,
            kernel_class=ScaledRBFKernel,
            y_std=0,
            mean_class=gpytorch.means.ZeroMean,
            likelihood_class=DirichletKernelClassifierLikelihood,
            likelihood_kwargs={"learn_alpha": False, "alpha": 5},
            marginal_log_likelihood_class=GenericExactMarginalLogLikelihood,
            optim_kwargs={"lr": 0.1, "early_stop_patience": 5},
            rng=self.rng,
        )
        gp.fit(1)


if __name__ == "__main__":
    unittest.main()
