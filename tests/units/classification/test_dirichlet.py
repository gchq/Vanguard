"""
Tests for the DirichletMulticlassClassification decorator.
"""

from unittest import TestCase

import numpy as np
import torch.testing
from gpytorch.likelihoods import DirichletClassificationLikelihood
from torch import Tensor

from vanguard.base.posteriors import Posterior
from vanguard.classification import DirichletMulticlassClassification
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.uncertainty import GaussianUncertaintyGPController
from vanguard.vanilla import GaussianGPController

from ...cases import flaky
from .case import BatchScaledMean, BatchScaledRBFKernel, ClassificationTestCase


@DirichletMulticlassClassification(num_classes=4, ignore_methods=("__init__",))
class DirichletMulticlassClassifier(GaussianGPController):
    """A simple Dirichlet multiclass classifier."""


class MulticlassTests(ClassificationTestCase):
    """
    Tests for multiclass classification.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.dataset = MulticlassGaussianClassificationDataset(num_train_points=150, num_test_points=100, num_classes=4)
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
        )

    @flaky
    def test_predictions(self) -> None:
        """Predictions should be close to the values from the test data."""
        self.controller.fit(10)
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.3)

    def test_illegal_likelihood_class(self) -> None:
        """Test that when an incorrect likelihood class is given, an appropriate exception is raised."""

        class IllegalLikelihoodClass:
            pass

        with self.assertRaises(ValueError) as ctx:
            __ = DirichletMulticlassClassifier(
                self.dataset.train_x,
                self.dataset.train_y,
                kernel_class=BatchScaledRBFKernel,
                y_std=0,
                likelihood_class=IllegalLikelihoodClass,
            )

        self.assertEqual(
            "The class passed to `likelihood_class` must be a subclass "
            f"of {DirichletClassificationLikelihood.__name__} for multiclass classification.",
            ctx.exception.args[0],
        )


class DirichletMulticlassFuzzyTests(ClassificationTestCase):
    """
    Tests for fuzzy dirichlet multiclass classification.
    """

    @flaky
    def test_fuzzy_predictions_monte_carlo(self) -> None:
        """Predictions should be close to the values from the test data."""
        dataset = MulticlassGaussianClassificationDataset(num_train_points=60, num_test_points=20, num_classes=4)
        test_x_std = 0.005
        test_x = np.random.normal(dataset.test_x, scale=test_x_std)

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
        )
        controller.fit(10)

        predictions, _ = controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(dataset.test_y, predictions, delta=0.5)

    @flaky
    def test_fuzzy_predictions_uncertainty(self) -> None:
        """Predictions should be close to the values from the test data."""
        dataset = MulticlassGaussianClassificationDataset(num_train_points=60, num_test_points=20, num_classes=4)

        train_x_std = test_x_std = 0.005
        train_x = np.random.normal(dataset.train_x, scale=train_x_std)
        test_x = np.random.normal(dataset.test_x, scale=test_x_std)

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
        )
        controller.fit(10)

        predictions, _ = controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(dataset.test_y, predictions, delta=0.4)


class InnerTests(TestCase):
    def setUp(self):
        """Set up data shared across tests."""
        self.rng = np.random.default_rng(1234)

    def test_posterior_from_mean_and_covariance(self):
        """Check that the wrapped class's posterior class transposes its mean matrix."""
        unwrapped_class = GaussianGPController
        wrapped_class = DirichletMulticlassClassification(num_classes=4, ignore_methods=("__init__",))(unwrapped_class)

        mean: Tensor = torch.tensor(self.rng.normal(size=(3,)))
        random_matrix: Tensor = torch.tensor(self.rng.normal(size=(3, 3)))
        cov = random_matrix @ random_matrix.T  # to ensure positive semidefinite

        unwrapped_posterior: Posterior = unwrapped_class.posterior_class.from_mean_and_covariance(mean, cov)
        wrapped_posterior: Posterior = wrapped_class.posterior_class.from_mean_and_covariance(mean, cov)

        torch.testing.assert_close(unwrapped_posterior.distribution.mean.T, wrapped_posterior.distribution.mean)
        torch.testing.assert_close(
            unwrapped_posterior.distribution.covariance_matrix, wrapped_posterior.distribution.covariance_matrix
        )
