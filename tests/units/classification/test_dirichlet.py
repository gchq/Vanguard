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
Tests for the DirichletMulticlassClassification decorator.
"""

from unittest.mock import Mock, patch

import torch
from gpytorch.likelihoods import DirichletClassificationLikelihood

from vanguard.base.posteriors import Posterior
from vanguard.classification import DirichletMulticlassClassification
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.uncertainty import GaussianUncertaintyGPController
from vanguard.vanilla import GaussianGPController

from ...cases import get_default_rng_override_seed
from .case import BatchScaledMean, BatchScaledRBFKernel, ClassificationTestCase

NUM_CLASSES = 4


@DirichletMulticlassClassification(num_classes=NUM_CLASSES, ignore_methods=("__init__",))
class DirichletMulticlassClassifier(GaussianGPController):
    """A simple Dirichlet multiclass classifier."""


class MulticlassTests(ClassificationTestCase):
    """
    Tests for multiclass classification.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        # Fails with seed 1234
        self.rng = get_default_rng_override_seed(12345)
        self.dataset = MulticlassGaussianClassificationDataset(
            num_train_points=150, num_test_points=100, num_classes=4, rng=self.rng
        )
        self.controller = DirichletMulticlassClassifier(
            self.dataset.train_x,
            self.dataset.train_y,
            y_std=0,
            mean_class=BatchScaledMean,
            kernel_class=BatchScaledRBFKernel,
            likelihood_class=DirichletClassificationLikelihood,
            likelihood_kwargs={"alpha_epsilon": 0.3, "learn_additional_noise": True},
            optim_kwargs={"lr": 0.05},
            kernel_kwargs={"batch_shape": 4},
            mean_kwargs={"batch_shape": 4},
            rng=self.rng,
        )

    def test_predictions(self) -> None:
        """Predict on a test dataset, and check the predictions are reasonably accurate."""
        self.controller.fit(10)
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.3)

    def test_classify_points_num_samples(self):
        """Test that the `n_posterior_samples` keyword argument is handled correctly when classifying points."""
        n_samples = 123

        self.controller.fit(10)
        mock_posterior = Mock(spec=Posterior)
        # pylint: disable-next=protected-access
        mock_posterior._tensor_sample = Mock(
            side_effect=lambda size: torch.ones((*size, self.dataset.test_x.shape[1], NUM_CLASSES))
        )
        with patch.object(GaussianGPController, "posterior_over_point", return_value=mock_posterior):
            self.controller.classify_points(self.dataset.test_x, n_posterior_samples=n_samples)
        # pylint: disable-next=protected-access
        mock_posterior._tensor_sample.assert_called_once_with(torch.Size((n_samples,)))

    def test_classify_fuzzy_points_num_samples(self):
        """Test that the `n_posterior_samples` keyword argument is handled correctly when classifying fuzzy points."""
        n_samples = 123

        self.controller.fit(10)
        mock_posterior = Mock(spec=Posterior)
        # pylint: disable-next=protected-access
        mock_posterior._tensor_sample_condensed = Mock(
            side_effect=lambda size: torch.ones((*size, self.dataset.test_x.shape[1], NUM_CLASSES))
        )
        with patch.object(GaussianGPController, "posterior_over_fuzzy_point", return_value=mock_posterior):
            self.controller.classify_fuzzy_points(
                self.dataset.test_x, self.dataset.test_x_std, n_posterior_samples=n_samples
            )
        # pylint: disable-next=protected-access
        mock_posterior._tensor_sample_condensed.assert_called_once_with(torch.Size((n_samples,)))

    def test_illegal_likelihood_class(self) -> None:
        """Test that when an incorrect `likelihood_class` is given, an appropriate exception is raised."""

        class IllegalLikelihoodClass:
            pass

        with self.assertRaises(ValueError) as ctx:
            __ = DirichletMulticlassClassifier(
                self.dataset.train_x,
                self.dataset.train_y,
                kernel_class=BatchScaledRBFKernel,
                y_std=0,
                likelihood_class=IllegalLikelihoodClass,
                rng=self.rng,
            )

        self.assertEqual(
            "The class passed to `likelihood_class` must be a subclass "
            f"of {DirichletClassificationLikelihood.__name__} for multiclass classification.",
            ctx.exception.args[0],
        )


class DirichletMulticlassFuzzyTests(ClassificationTestCase):
    """
    Tests for fuzzy Dirichlet multiclass classification.
    """

    def setUp(self) -> None:
        """Set up data shared across tests."""
        # test_fuzzy_predictions_uncertainty() fails with seed 1234
        self.rng = get_default_rng_override_seed(12345)

    def test_fuzzy_predictions_monte_carlo(self) -> None:
        """
        Predict on a noisy test dataset, and check the predictions are reasonably accurate.

        In this test, the training inputs have no noise applied, but the test inputs do.

        Note that we ignore the `certainties` output here.
        """
        dataset = MulticlassGaussianClassificationDataset(
            num_train_points=60, num_test_points=20, num_classes=4, rng=self.rng
        )
        test_x_std = 0.005
        test_x = self.rng.normal(dataset.test_x, scale=test_x_std)

        controller = DirichletMulticlassClassifier(
            dataset.train_x,
            dataset.train_y,
            y_std=0,
            mean_class=BatchScaledMean,
            kernel_class=BatchScaledRBFKernel,
            likelihood_class=DirichletClassificationLikelihood,
            likelihood_kwargs={"alpha_epsilon": 0.3, "learn_additional_noise": True},
            optim_kwargs={"lr": 0.05},
            kernel_kwargs={"batch_shape": 4},
            mean_kwargs={"batch_shape": 4},
            rng=self.rng,
        )
        controller.fit(10)

        predictions, _ = controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(dataset.test_y, predictions, delta=0.5)

    def test_fuzzy_predictions_uncertainty(self) -> None:
        """
        Predict on a noisy test dataset, and check the predictions are reasonably accurate.

        In this test, the training and test inputs have the same level of noise applied, and we use
        `GaussianUncertaintyGPController` as a base class for the controller to allow us to handle the noise.

        Note that we ignore the `certainties` output here.
        """
        dataset = MulticlassGaussianClassificationDataset(
            num_train_points=60, num_test_points=20, num_classes=4, rng=self.rng
        )

        train_x_std = test_x_std = 0.005
        train_x = self.rng.normal(dataset.train_x, scale=train_x_std)
        test_x = self.rng.normal(dataset.test_x, scale=test_x_std)

        @DirichletMulticlassClassification(num_classes=4, ignore_all=True)
        class UncertaintyDirichletMulticlassClassifier(GaussianUncertaintyGPController):
            """A simple uncertain multiclass classifier."""

        controller = UncertaintyDirichletMulticlassClassifier(
            train_x,
            train_x_std,
            dataset.train_y,
            mean_class=BatchScaledMean,
            kernel_class=BatchScaledRBFKernel,
            y_std=0,
            likelihood_class=DirichletClassificationLikelihood,
            likelihood_kwargs={"alpha_epsilon": 0.3, "learn_additional_noise": True},
            optim_kwargs={"lr": 0.05},
            kernel_kwargs={"batch_shape": 4},
            mean_kwargs={"batch_shape": 4},
            rng=self.rng,
        )
        controller.fit(10)

        predictions, _ = controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(dataset.test_y, predictions, delta=0.4)
