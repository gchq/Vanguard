"""
Tests for the DirichletKernelMulticlassClassification decorator.
"""
from unittest import skip

from gpytorch import kernels, means

from vanguard.classification.kernel import DirichletKernelMulticlassClassification
from vanguard.classification.likelihoods import DirichletKernelClassifierLikelihood, GenericExactMarginalLogLikelihood
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
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
    def setUp(self) -> None:
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
    def test_predictions(self) -> None:
        """Predictions should be close to the values from the test data."""
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.3)

    @skip("Currently hangs - gets stuck in an infinite loop in MonteCarloPosteriorCollection._yield_posteriors")
    @flaky
    def test_fuzzy_predictions(self) -> None:
        """Predictions should be close to the values from the test data."""
        test_x_std = 0.005
        predictions, _ = self.controller.classify_fuzzy_points(self.dataset.test_x, test_x_std)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.3)


    def test_illegal_likelihood_class(self) -> None:
        """Test that when an incorrect likelihood class is given, an appropriate exception is raised."""
        class IllegalLikelihoodClass:
            pass

        with self.assertRaises(ValueError) as ctx:
            __ = MulticlassGaussianClassifier(self.dataset.train_x, self.dataset.train_y,
                                              mean_class=means.ZeroMean, kernel_class=kernels.RBFKernel,
                                              y_std=0, likelihood_class=IllegalLikelihoodClass)

        self.assertEqual(
            "The class passed to `likelihood_class` must be a subclass of "
            f"{DirichletKernelClassifierLikelihood.__name__}.",
            # TODO should this say "multiclass classification" instead?
            ctx.exception.args[0]
        )
