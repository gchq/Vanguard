"""
Tests for the DirichletMulticlassClassification decorator.
"""
import numpy as np
from gpytorch.likelihoods import DirichletClassificationLikelihood

from vanguard.classification import DirichletMulticlassClassification
from vanguard.datasets.classification import \
    MulticlassGaussianClassificationDataset
from vanguard.uncertainty import GaussianUncertaintyGPController
from vanguard.vanilla import GaussianGPController

from ...cases import flaky
from .case import BatchScaledMean, BatchScaledRBFKernel, ClassificationTestCase


@DirichletMulticlassClassification(num_classes=4, ignore_methods=("__init__",))
class DirichletMulticlassClassifier(GaussianGPController):
    """A simple Dirichlet multiclass classifier."""
    pass


class MulticlassTests(ClassificationTestCase):
    """
    Tests for multiclass classification.
    """
    def setUp(self):
        """Code to run before each test."""
        self.dataset = MulticlassGaussianClassificationDataset(num_train_points=150, num_test_points=100,
                                                               num_classes=4)
        self.controller = DirichletMulticlassClassifier(self.dataset.train_x, self.dataset.train_y, y_std=0,
                                                        mean_class=BatchScaledMean, kernel_class=BatchScaledRBFKernel,
                                                        likelihood_class=DirichletClassificationLikelihood,
                                                        likelihood_kwargs={"alpha_epsilon": 0.3,
                                                                           "learn_additional_noise": True},
                                                        optim_kwargs={"lr": 0.05}, kernel_kwargs={"batch_shape": 4},
                                                        mean_kwargs={"batch_shape": 4})
        self.controller.fit(100)

    @flaky
    def test_predictions(self):
        """Predictions should be close to the values from the test data."""
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.3)


class DirichletMulticlassFuzzyTests(ClassificationTestCase):
    """
    Tests for fuzzy dirichlet multiclass classification.
    """
    @flaky
    def test_fuzzy_predictions_monte_carlo(self):
        """Predictions should be close to the values from the test data."""
        self.dataset = MulticlassGaussianClassificationDataset(num_train_points=150, num_test_points=20,
                                                               num_classes=4)
        test_x_std = 0.005
        test_x = np.random.normal(self.dataset.test_x, scale=test_x_std)

        self.controller = DirichletMulticlassClassifier(self.dataset.train_x, self.dataset.train_y, y_std=0,
                                                        mean_class=BatchScaledMean, kernel_class=BatchScaledRBFKernel,
                                                        likelihood_class=DirichletClassificationLikelihood,
                                                        likelihood_kwargs={"alpha_epsilon": 0.3,
                                                                           "learn_additional_noise": True},
                                                        optim_kwargs={"lr": 0.05}, kernel_kwargs={"batch_shape": 4},
                                                        mean_kwargs={"batch_shape": 4})
        self.controller.fit(10)

        predictions, _ = self.controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.5)

    @flaky
    def test_fuzzy_predictions_uncertainty(self):
        """Predictions should be close to the values from the test data."""
        self.dataset = MulticlassGaussianClassificationDataset(num_train_points=150, num_test_points=100,
                                                               num_classes=4)

        train_x_std = test_x_std = 0.005
        train_x = np.random.normal(self.dataset.train_x, scale=train_x_std)
        test_x = np.random.normal(self.dataset.test_x, scale=test_x_std)

        @DirichletMulticlassClassification(num_classes=4, ignore_all=True)
        class UncertaintyDirichletMulticlassClassifier(GaussianUncertaintyGPController):
            """A simple uncertain multiclass classifier."""
            pass

        self.controller = UncertaintyDirichletMulticlassClassifier(train_x, train_x_std, self.dataset.train_y,
                                                                   mean_class=BatchScaledMean,
                                                                   kernel_class=BatchScaledRBFKernel, y_std=0,
                                                                   likelihood_class=DirichletClassificationLikelihood,
                                                                   likelihood_kwargs={"alpha_epsilon": 0.3,
                                                                                      "learn_additional_noise": True},
                                                                   optim_kwargs={"lr": 0.05},
                                                                   kernel_kwargs={"batch_shape": 4},
                                                                   mean_kwargs={"batch_shape": 4})
        self.controller.fit(50)

        predictions, _ = self.controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.4)
