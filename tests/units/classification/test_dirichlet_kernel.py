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
Tests for the DirichletKernelMulticlassClassification decorator.
"""

from unittest import skip

import pytest
import torch
from gpytorch import kernels, means

from vanguard.classification.kernel import DirichletKernelMulticlassClassification
from vanguard.classification.likelihoods import DirichletKernelClassifierLikelihood, GenericExactMarginalLogLikelihood
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.vanilla import GaussianGPController

from ...cases import get_default_rng
from .case import ClassificationTestCase


@DirichletKernelMulticlassClassification(num_classes=4, ignore_methods=("__init__",))
class MulticlassGaussianClassifier(GaussianGPController):
    pass


class MulticlassTests(ClassificationTestCase):
    """
    Tests for multiclass classification.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.rng = get_default_rng()
        self.dataset = MulticlassGaussianClassificationDataset(
            num_train_points=150, num_test_points=100, num_classes=4, rng=self.rng
        )
        self.controller = MulticlassGaussianClassifier(
            self.dataset.train_x,
            self.dataset.train_y,
            y_std=0.0,
            mean_class=means.ZeroMean,
            kernel_class=kernels.RBFKernel,
            likelihood_class=DirichletKernelClassifierLikelihood,
            optim_kwargs={"lr": 0.05},
            marginal_log_likelihood_class=GenericExactMarginalLogLikelihood,
            rng=self.rng,
        )
        self.controller.fit(10)

    def test_get_posterior_over_point_in_eval_mode(self) -> None:
        """
        Check the dimensions of mean and covariance objects returned by _get_posterior_over_point_in_eval_mode().
        """
        num_test_points = self.dataset.test_x.shape[0]
        num_classes = self.dataset.num_classes

        # pylint: disable-next=protected-access
        prediction_output = self.controller._get_posterior_over_point_in_eval_mode(self.dataset.test_x)
        mean, covar = prediction_output.distribution.mean, prediction_output.distribution.covariance_matrix
        self.assertEqual(mean.shape, torch.Size([num_test_points, num_classes]))
        self.assertEqual(covar.shape, torch.Size([num_classes, num_classes, num_test_points, num_test_points]))

    def test_predictions(self) -> None:
        """Predict on a test dataset, and check the predictions are reasonably accurate."""
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.4)

    def test_get_posterior_over_fuzzy_point_in_eval_mode(self) -> None:
        """
        Check the dimensions of mean and covariance objects returned by _get_posterior_over_fuzzy_point_in_eval_mode().
        """
        test_x_std = 0.005
        num_test_points = self.dataset.test_x.shape[0]
        num_classes = self.dataset.num_classes
        default_group_size = 100  # in infinite_x_samples()

        # pylint: disable-next=protected-access
        prediction_output = self.controller._get_posterior_over_fuzzy_point_in_eval_mode(
            self.dataset.test_x, test_x_std
        )
        mean, covar = prediction_output.distribution.mean, prediction_output.distribution.covariance_matrix
        self.assertEqual(mean.shape, torch.Size([default_group_size, num_test_points, num_classes]))
        self.assertEqual(
            covar.shape, torch.Size([default_group_size, num_classes, num_classes, num_test_points, num_test_points])
        )

    # TODO: When using the original code for classify_fuzzy_points, the test below fails, as the distribution
    #  covariance_matrix is an unexpected shape.
    # https://github.com/gchq/Vanguard/issues/288
    @skip
    def test_fuzzy_predictions(self) -> None:
        """
        Predict on a noisy test dataset, and check the predictions are reasonably accurate.

        In this test, the training inputs have no noise applied, but the test inputs do.

        Note that we ignore the `certainties` output here.
        """
        test_x_std = 0.005
        test_x = self.rng.normal(self.dataset.test_x, test_x_std)
        predictions, _ = self.controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.4)

    def test_fuzzy_predictions_not_implemented(self):
        """Check that calling classify_fuzzy_points raises an error informing the user that it's not supported."""
        with pytest.raises(NotImplementedError, match="Fuzzy classification is not supported"):
            self.controller.classify_fuzzy_points(self.dataset.test_x, 0.1)

    def test_illegal_likelihood_class(self) -> None:
        """Test that when an incorrect likelihood class is given, an appropriate exception is raised."""

        class IllegalLikelihoodClass:
            pass

        with self.assertRaises(ValueError) as ctx:
            __ = MulticlassGaussianClassifier(
                self.dataset.train_x,
                self.dataset.train_y,
                mean_class=means.ZeroMean,
                kernel_class=kernels.RBFKernel,
                y_std=0.0,
                likelihood_class=IllegalLikelihoodClass,
                rng=self.rng,
            )

        self.assertEqual(
            "The class passed to `likelihood_class` must be a subclass of "
            f"{DirichletKernelClassifierLikelihood.__name__}.",
            ctx.exception.args[0],
        )
