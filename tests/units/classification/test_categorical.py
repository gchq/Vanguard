"""
Tests for the CategoricalClassification decorator.
"""

import unittest

import numpy as np
import sklearn
from gpytorch.mlls import VariationalELBO

from vanguard.classification import CategoricalClassification
from vanguard.classification.likelihoods import MultitaskBernoulliLikelihood, SoftmaxLikelihood
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.multitask import Multitask
from vanguard.uncertainty import GaussianUncertaintyGPController
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference

from ...cases import flaky
from .case import BatchScaledMean, ClassificationTestCase

one_hot = sklearn.preprocessing.LabelBinarizer().fit_transform

NUM_LATENTS = 10


@CategoricalClassification(num_classes=4, ignore_all=True)
@Multitask(num_tasks=4, ignore_all=True)
@VariationalInference(ignore_all=True)
class MultitaskBernoulliClassifier(GaussianGPController):
    """A simple multi-class Bernoulli classifier."""


@CategoricalClassification(num_classes=4, ignore_all=True)
@Multitask(num_tasks=6, lmc_dimension=NUM_LATENTS, ignore_all=True)
@VariationalInference(ignore_all=True)
class SoftmaxLMCClassifier(GaussianGPController):
    """A simple multi-class classifier with LMC."""


@CategoricalClassification(num_classes=4, ignore_all=True)
@Multitask(num_tasks=NUM_LATENTS, lmc_dimension=None, ignore_all=True)
@VariationalInference(ignore_all=True)
class SoftmaxClassifier(GaussianGPController):
    """A simple multi-class classifier without LMC."""


@CategoricalClassification(num_classes=4, ignore_all=True)
@Multitask(num_tasks=5, ignore_all=True)
@VariationalInference(ignore_all=True)
class MultitaskBernoulliClassifierWrongNumberOfTasks(GaussianGPController):
    """An incorrectly configured multi-class classifier."""


class MulticlassTests(ClassificationTestCase):
    """
    Tests for binary classification.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.dataset = MulticlassGaussianClassificationDataset(num_train_points=60, num_test_points=20, num_classes=4)
        self.controller = MultitaskBernoulliClassifier(
            self.dataset.train_x,
            one_hot(self.dataset.train_y),
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=MultitaskBernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
        )
        self.controller.fit(10)

    @flaky
    def test_predictions(self) -> None:
        """Predictions should be close to the values from the test data."""
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.4)


class MulticlassFuzzyTests(ClassificationTestCase):
    """
    Tests for fuzzy multiclass classification.
    """

    # TODO: Seems too flaky on 3.8 and 3.9 but reliable on 3.12, especially when delta=0.5.
    # https://github.com/gchq/Vanguard/issues/128
    @flaky
    def test_fuzzy_predictions_monte_carlo(self) -> None:
        """Predictions should be close to the values from the test data."""
        dataset = MulticlassGaussianClassificationDataset(num_train_points=60, num_test_points=20, num_classes=4)
        test_x_std = 0.005
        test_x = np.random.normal(dataset.test_x, scale=test_x_std)

        controller = MultitaskBernoulliClassifier(
            dataset.train_x,
            one_hot(dataset.train_y),
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=MultitaskBernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
        )
        controller.fit(10)

        predictions, _ = controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(dataset.test_y, predictions, delta=0.5)

    def test_fuzzy_predictions_uncertainty(self) -> None:
        """Predictions should be close to the values from the test data."""
        dataset = MulticlassGaussianClassificationDataset(num_train_points=60, num_test_points=20, num_classes=4)
        train_x_std = test_x_std = 0.005
        train_x = np.random.normal(dataset.train_x, scale=train_x_std)
        test_x = np.random.normal(dataset.test_x, scale=test_x_std)

        @CategoricalClassification(num_classes=4, ignore_all=True)
        @Multitask(num_tasks=4, ignore_all=True)
        @VariationalInference(ignore_all=True)
        class UncertaintyMultitaskBernoulliClassifier(GaussianUncertaintyGPController):
            """An uncertain multitask classifier."""

        controller = UncertaintyMultitaskBernoulliClassifier(
            train_x,
            train_x_std,
            one_hot(dataset.train_y),
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=MultitaskBernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
        )
        controller.fit(10)

        predictions, _ = controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(dataset.test_y, predictions, delta=0.5)


class SoftmaxLMCTests(unittest.TestCase):
    """
    Tests for softmax multi-class classification with LMC
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.dataset = MulticlassGaussianClassificationDataset(num_train_points=100, num_test_points=500, num_classes=4)

        self.controller = SoftmaxLMCClassifier(
            self.dataset.train_x,
            self.dataset.train_y,
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=SoftmaxLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
        )

    def test_fitting(self) -> None:
        """Test that fitting is possible."""
        self.controller.fit(1)


class SoftmaxTests(unittest.TestCase):
    """
    Tests for softmax multi-class classification without LMC.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.dataset = MulticlassGaussianClassificationDataset(num_train_points=100, num_test_points=500, num_classes=4)

        self.controller = SoftmaxClassifier(
            self.dataset.train_x,
            self.dataset.train_y,
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=SoftmaxLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
        )

    def test_fitting(self) -> None:
        """Test that fitting is possible."""
        self.controller.fit(1)

    def test_fitting_with_mismatch_mean_errors(self) -> None:
        """Test for error when creating controller with a mean of the wrong shape."""
        with self.assertRaises(TypeError):
            self.controller = SoftmaxClassifier(
                self.dataset.train_x,
                self.dataset.train_y,
                kernel_class=ScaledRBFKernel,
                mean_class=BatchScaledMean,
                y_std=0,
                likelihood_class=SoftmaxLikelihood,
                marginal_log_likelihood_class=VariationalELBO,
                mean_kwargs={"batch_shape": NUM_LATENTS + 2},
            )


class MultitaskBernoulliClassifierTests(unittest.TestCase):
    """
    Tests for softmax multi-class classification with LMC
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.dataset = MulticlassGaussianClassificationDataset(num_train_points=100, num_test_points=500, num_classes=4)

        self.controller = MultitaskBernoulliClassifier(
            self.dataset.train_x,
            one_hot(self.dataset.train_y),
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=MultitaskBernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
        )

    def test_fitting(self) -> None:
        """Test that fitting is possible."""
        self.controller.fit(1)
