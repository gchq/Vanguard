"""
Tests for the DirichletKernelMulticlassClassification decorator.
"""
from gpytorch import kernels, means

from vanguard.classification.kernel import \
    DirichletKernelMulticlassClassification
from vanguard.classification.likelihoods import (
    DirichletKernelClassifierLikelihood, GenericExactMarginalLogLikelihood)
from vanguard.datasets.classification import \
    MulticlassGaussianClassificationDataset
from vanguard.vanilla import GaussianGPController

from ...cases import flaky
from .case import ClassificationTestCase


@DirichletKernelMulticlassClassification(num_classes=4, ignore_methods=("__init__",))
class MulticlassGaussianClassifier(GaussianGPController):
    pass


class MulticlassTests(ClassificationTestCase):
    """
    Tests for multiclass classification.
    """
    def setUp(self):
        """Code to run before each test."""
        self.dataset = MulticlassGaussianClassificationDataset(num_train_points=150, num_test_points=100,
                                                               num_classes=4)
        self.controller = MulticlassGaussianClassifier(self.dataset.train_x, self.dataset.train_y, y_std=0,
                                                       mean_class=means.ZeroMean, kernel_class=kernels.RBFKernel,
                                                       likelihood_class=DirichletKernelClassifierLikelihood,
                                                       optim_kwargs={"lr": 0.05},
                                                       marginal_log_likelihood_class=GenericExactMarginalLogLikelihood)
        self.controller.fit(100)

    @flaky
    def test_predictions(self):
        """Predictions should be close to the values from the test data."""
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.3)
