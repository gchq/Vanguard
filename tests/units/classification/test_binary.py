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
Tests for the BinaryClassification decorator.
"""

import torch
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO

from vanguard.classification import BinaryClassification
from vanguard.datasets.classification import BinaryStripeClassificationDataset
from vanguard.kernels import PeriodicRBFKernel
from vanguard.uncertainty import GaussianUncertaintyGPController
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference

from ...cases import get_default_rng_override_seed, get_default_torch_rng
from .case import ClassificationTestCase


@BinaryClassification(ignore_methods=("__init__", "_predictive_likelihood", "_fuzzy_predictive_likelihood"))
@VariationalInference(ignore_methods=("__init__",))
class BinaryClassifier(GaussianGPController):
    """A simple binary classifier."""


class BinaryTests(ClassificationTestCase):
    """
    Tests for binary classification.
    """

    def setUp(self) -> None:
        """Set up data shared between tests."""
        self.rng = get_default_rng_override_seed(123_456)  # Fails on Windows with 1234; fails on Linux with 12345
        self.dataset = BinaryStripeClassificationDataset(num_train_points=100, num_test_points=200, rng=self.rng)
        self.controller = BinaryClassifier(
            self.dataset.train_x,
            self.dataset.train_y,
            kernel_class=PeriodicRBFKernel,
            y_std=0.0,
            likelihood_class=BernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            rng=self.rng,
        )

    def test_predictions(self) -> None:
        """Predict on a test dataset, and check the predictions are reasonably accurate."""
        self.controller.fit(20)
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y.squeeze(), predictions, delta=0.3)

    def test_illegal_likelihood_class(self) -> None:
        """Test that when an incorrect likelihood class is given, an appropriate exception is raised."""

        class IllegalLikelihoodClass:
            pass

        with self.assertRaises(ValueError) as ctx:
            __ = BinaryClassifier(
                self.dataset.train_x,
                self.dataset.train_y,
                kernel_class=PeriodicRBFKernel,
                y_std=0.0,
                likelihood_class=IllegalLikelihoodClass,
                marginal_log_likelihood_class=VariationalELBO,
                rng=self.rng,
            )

        self.assertEqual(
            "The class passed to `likelihood_class` must be a subclass "
            f"of {BernoulliLikelihood.__name__} for binary classification.",
            ctx.exception.args[0],
        )

    def test_closed_methods(self):
        """
        Test that the `ClassificationMixin` has correctly closed the prediction methods of the underlying controller.

        In particular, we test that we get an appropriate error message directing us towards the corresponding
        classification method instead.
        """
        cases = [
            ((lambda: self.controller.posterior_over_point(1.0)), "classify_points"),
            ((lambda: self.controller.posterior_over_fuzzy_point(1.0, 1.0)), "classify_fuzzy_points"),
            ((lambda: self.controller.predictive_likelihood(1.0)), "classify_points"),
            ((lambda: self.controller.fuzzy_predictive_likelihood(1.0, 1.0)), "classify_fuzzy_points"),
        ]

        for call_method, alternative_method in cases:
            with self.subTest():
                with self.assertRaises(TypeError) as ctx:
                    call_method()
                self.assertEqual(f"The '{alternative_method}' method should be used instead.", ctx.exception.args[0])


class BinaryFuzzyTests(ClassificationTestCase):
    """
    Tests for fuzzy binary classification.
    """

    def setUp(self) -> None:
        """Set up data shared between tests."""
        self.rng = get_default_rng_override_seed(123_456)  # Fails on Windows with 1234; fails on Linux with 12345
        self.torch_rng = get_default_torch_rng()

    def test_fuzzy_predictions_monte_carlo(self) -> None:
        """
        Predict on a noisy test dataset, and check the predictions are reasonably accurate.

        In this test, the training inputs have no noise applied, but the test inputs do.

        Note that we ignore the `certainties` output here.
        """
        dataset = BinaryStripeClassificationDataset(num_train_points=50, num_test_points=40, rng=self.rng)
        test_x_std = 0.005
        test_x = torch.normal(dataset.test_x, std=test_x_std, generator=self.torch_rng)

        controller = BinaryClassifier(
            dataset.train_x,
            dataset.train_y,
            kernel_class=PeriodicRBFKernel,
            y_std=0.0,
            likelihood_class=BernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            rng=self.rng,
        )
        controller.fit(20)

        predictions, _ = controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(dataset.test_y.squeeze(), predictions, delta=0.25)

    def test_fuzzy_predictions_uncertainty(self) -> None:
        """
        Predict on a noisy test dataset, and check the predictions are reasonably accurate.

        In this test, the training and test inputs have the same level of noise applied, and we use
        `GaussianUncertaintyGPController` as a base class for the controller to allow us to handle the noise.

        Note that we ignore the `certainties` output here.
        """
        dataset = BinaryStripeClassificationDataset(50, 40, rng=self.rng)
        train_x_std = test_x_std = 0.005
        train_x = torch.normal(dataset.train_x, std=train_x_std, generator=self.torch_rng)
        test_x = torch.normal(dataset.test_x, std=test_x_std, generator=self.torch_rng).reshape(-1, 1)

        @BinaryClassification(ignore_all=True)
        @VariationalInference(ignore_all=True)
        class UncertaintyBinaryClassifier(GaussianUncertaintyGPController):
            """A simple binary classifier."""

        controller = UncertaintyBinaryClassifier(
            train_x,
            train_x_std,
            dataset.train_y,
            kernel_class=PeriodicRBFKernel,
            y_std=0.0,
            likelihood_class=BernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            rng=self.rng,
        )
        controller.fit(30)

        predictions, _ = controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(dataset.test_y.squeeze(), predictions, delta=0.3)
