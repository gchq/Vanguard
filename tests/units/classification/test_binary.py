"""
Tests for the BinaryClassification decorator.
"""

from unittest import expectedFailure

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


class BinaryTests(ClassificationTestCase):
    """
    Tests for binary classification.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.dataset = BinaryStripeClassificationDataset(num_train_points=100, num_test_points=200)
        self.controller = BinaryClassifier(
            self.dataset.train_x,
            self.dataset.train_y,
            kernel_class=PeriodicRBFKernel,
            y_std=0,
            likelihood_class=BernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
        )
        self.controller.fit(100)

    @flaky
    def test_predictions(self) -> None:
        """Predictions should be close to the values from the test data."""
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.05)

    def test_illegal_likelihood_class(self) -> None:
        """Test that when an incorrect likelihood class is given, an appropriate exception is raised."""

        class IllegalLikelihoodClass:
            pass

        with self.assertRaises(ValueError) as ctx:
            __ = BinaryClassifier(
                self.dataset.train_x,
                self.dataset.train_y,
                kernel_class=PeriodicRBFKernel,
                y_std=0,
                likelihood_class=IllegalLikelihoodClass,
                marginal_log_likelihood_class=VariationalELBO,
            )

        self.assertEqual(
            "The class passed to `likelihood_class` must be a subclass "
            f"of {BernoulliLikelihood.__name__} for binary classification.",
            ctx.exception.args[0],
        )

    @expectedFailure  # TODO: These tests currently fail. Find out why ClassificationMixin isn't working properly.
    def test_closed_methods(self):
        """Test that the ClassificationMixin has correctly closed the prediction methods of the underlying controller"""
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

    @flaky
    def test_fuzzy_predictions_monte_carlo(self) -> None:
        """Predictions should be close to the values from the test data."""
        dataset = BinaryStripeClassificationDataset(num_train_points=100, num_test_points=50)
        test_x_std = 0.005
        test_x = np.random.normal(dataset.test_x, scale=test_x_std)

        controller = BinaryClassifier(
            dataset.train_x,
            dataset.train_y,
            kernel_class=PeriodicRBFKernel,
            y_std=0,
            likelihood_class=BernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
        )
        controller.fit(100)

        predictions, _ = controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(dataset.test_y, predictions, delta=0.1)

    @flaky
    def test_fuzzy_predictions_uncertainty(self) -> None:
        """Predictions should be close to the values from the test data."""
        dataset = BinaryStripeClassificationDataset(100, 50)
        train_x_std = test_x_std = 0.005
        train_x = np.random.normal(dataset.train_x, scale=train_x_std)
        test_x = np.random.normal(dataset.test_x, scale=test_x_std).reshape(-1, 1)

        @BinaryClassification(ignore_all=True)
        @VariationalInference(ignore_all=True)
        class UncertaintyBinaryClassifier(GaussianUncertaintyGPController):
            """A simple binary classifier."""

        controller = UncertaintyBinaryClassifier(
            train_x,
            train_x_std,
            dataset.train_y,
            kernel_class=PeriodicRBFKernel,
            y_std=0,
            likelihood_class=BernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
        )
        controller.fit(100)

        predictions, _ = controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(dataset.test_y, predictions, delta=0.1)
