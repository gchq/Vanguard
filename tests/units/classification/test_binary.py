"""
Tests for the BinaryClassification decorator.
"""
import numpy as np
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO

from vanguard.classification import BinaryClassification
from vanguard.datasets.classification import BinaryStripeClassificationDataset
from vanguard.kernels import PeriodicRBFKernel
from vanguard.uncertainty import GaussianUncertaintyGPController
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference

from ...cases import flaky
from .case import ClassificationTestCase


@BinaryClassification(ignore_methods=("__init__", "_predictive_likelihood", "_fuzzy_predictive_likelihood"))
@VariationalInference(ignore_methods=("__init__",))
class BinaryClassifier(GaussianGPController):
    """A simple binary classifier."""
    pass


class BinaryTests(ClassificationTestCase):
    """
    Tests for binary classification.
    """
    def setUp(self) -> None:
        """Code to run before each test."""
        self.dataset = BinaryStripeClassificationDataset(num_train_points=100, num_test_points=200)
        self.controller = BinaryClassifier(self.dataset.train_x, self.dataset.train_y, kernel_class=PeriodicRBFKernel,
                                           y_std=0, likelihood_class=BernoulliLikelihood,
                                           marginal_log_likelihood_class=VariationalELBO)
        self.controller.fit(100)

    @flaky
    def test_predictions(self) -> None:
        """Predictions should be close to the values from the test data."""
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.05)


class BinaryFuzzyTests(ClassificationTestCase):
    """
    Tests for fuzzy binary classification.
    """
    @flaky
    def test_fuzzy_predictions_monte_carlo(self) -> None:
        """Predictions should be close to the values from the test data."""
        self.dataset = BinaryStripeClassificationDataset(num_train_points=100, num_test_points=50)
        test_x_std = 0.005
        test_x = np.random.normal(self.dataset.test_x, scale=test_x_std)

        self.controller = BinaryClassifier(self.dataset.train_x, self.dataset.train_y, kernel_class=PeriodicRBFKernel,
                                           y_std=0, likelihood_class=BernoulliLikelihood,
                                           marginal_log_likelihood_class=VariationalELBO)
        self.controller.fit(100)

        predictions, _ = self.controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.1)

    @flaky
    def test_fuzzy_predictions_uncertainty(self) -> None:
        """Predictions should be close to the values from the test data."""
        self.dataset = BinaryStripeClassificationDataset(100, 50)
        train_x_std = test_x_std = 0.005
        train_x = np.random.normal(self.dataset.train_x, scale=train_x_std)
        test_x = np.random.normal(self.dataset.test_x, scale=test_x_std).reshape(-1, 1)

        @BinaryClassification(ignore_all=True)
        @VariationalInference(ignore_all=True)
        class UncertaintyBinaryClassifier(GaussianUncertaintyGPController):
            """A simple binary classifier."""
            pass

        self.controller = UncertaintyBinaryClassifier(train_x, train_x_std, self.dataset.train_y,
                                                      kernel_class=PeriodicRBFKernel, y_std=0,
                                                      likelihood_class=BernoulliLikelihood,
                                                      marginal_log_likelihood_class=VariationalELBO)
        self.controller.fit(100)

        predictions, _ = self.controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.1)
