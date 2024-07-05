"""
Tests for the CategoricalClassification decorator.
"""

from unittest import expectedFailure

import sklearn
import torch
from gpytorch.mlls import VariationalELBO

from vanguard.classification import CategoricalClassification
from vanguard.classification.likelihoods import MultitaskBernoulliLikelihood, SoftmaxLikelihood
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.multitask import Multitask
from vanguard.uncertainty import GaussianUncertaintyGPController
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference

from ...cases import get_default_rng
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
        self.rng = get_default_rng()
        self.dataset = MulticlassGaussianClassificationDataset(
            num_train_points=60, num_test_points=20, num_classes=4, rng=self.rng
        )
        self.controller = MultitaskBernoulliClassifier(
            self.dataset.train_x,
            one_hot(self.dataset.train_y),
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=MultitaskBernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            rng=self.rng,
        )

    def test_predictions(self) -> None:
        """Predict on a test dataset, and check the predictions are reasonably accurate."""
        self.controller.fit(10)
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.4)


class MulticlassFuzzyTests(ClassificationTestCase):
    """
    Tests for fuzzy multiclass classification.
    """

    def setUp(self) -> None:
        """Set up data shared between tests."""
        self.rng = get_default_rng()

    # TODO: Seems too flaky on 3.8 and 3.9 but reliable on 3.12, especially when delta=0.5.
    # https://github.com/gchq/Vanguard/issues/128
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

        controller = MultitaskBernoulliClassifier(
            dataset.train_x,
            one_hot(dataset.train_y),
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=MultitaskBernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
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
            rng=self.rng,
        )
        controller.fit(10)

        predictions, _ = controller.classify_fuzzy_points(test_x, test_x_std)
        self.assertPredictionsEqual(dataset.test_y, predictions, delta=0.5)


class SoftmaxLMCTests(ClassificationTestCase):
    """
    Tests for softmax multi-class classification with LMC.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.rng = get_default_rng()
        self.dataset = MulticlassGaussianClassificationDataset(
            num_train_points=60, num_test_points=20, num_classes=4, rng=self.rng
        )

        self.controller = SoftmaxLMCClassifier(
            self.dataset.train_x,
            self.dataset.train_y,
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=SoftmaxLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            rng=self.rng,
        )

    def test_predictions(self) -> None:
        """Predict on a test dataset, and check the predictions are reasonably accurate."""
        self.controller.fit(10)
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.4)


class SoftmaxTests(ClassificationTestCase):
    """
    Tests for softmax multi-class classification without LMC.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.rng = get_default_rng()
        self.dataset = MulticlassGaussianClassificationDataset(
            num_train_points=60, num_test_points=20, num_classes=4, rng=self.rng
        )

        self.controller = SoftmaxClassifier(
            self.dataset.train_x,
            self.dataset.train_y,
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=SoftmaxLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            rng=self.rng,
        )

    def test_predictions(self) -> None:
        """Predict on a test dataset, and check the predictions are reasonably accurate."""
        self.controller.fit(10)
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.4)

    # TODO: fails with the following error:
    #  RuntimeError: grad can be implicitly created only for scalar outputs
    # https://github.com/gchq/Vanguard/issues/290
    @expectedFailure
    def test_fitting_with_batch_shape(self) -> None:
        """Test that fitting is possible when the kwarg `batch_shape` is passed to the `BatchScaledMean` class."""
        controller = SoftmaxClassifier(
            self.dataset.train_x,
            self.dataset.train_y,
            kernel_class=ScaledRBFKernel,
            mean_class=BatchScaledMean,
            y_std=0,
            likelihood_class=SoftmaxLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            mean_kwargs={"batch_shape": torch.Size([NUM_LATENTS])},
            rng=self.rng,
        )

        controller.fit(1)

    def test_creating_with_invalid_mean_type_errors(self) -> None:
        """Test that creating a controller with a `batch_shape` of incorrect type raises an appropriate error."""
        with self.assertRaises(TypeError) as ctx:
            SoftmaxClassifier(
                self.dataset.train_x,
                self.dataset.train_y,
                kernel_class=ScaledRBFKernel,
                mean_class=BatchScaledMean,
                y_std=0,
                likelihood_class=SoftmaxLikelihood,
                marginal_log_likelihood_class=VariationalELBO,
                mean_kwargs={"batch_shape": NUM_LATENTS},
                rng=self.rng,
            )

        self.assertEqual(
            "Expected mean_kwargs['batch_shape'] to be of type `torch.Size`; got `int` instead",
            str(ctx.exception),
        )

    # TODO: This fails as there's no code to check for this error. Unsure whether it should be in __init__ or in fit().
    #  Adding code to check for it is difficult as it doesn't work even when the shape matches. See the linked issue
    #  which is blocking this.
    # https://github.com/gchq/Vanguard/issues/290
    @expectedFailure
    def test_creating_with_mismatched_mean_shape_errors(self) -> None:
        """Test that creating controller with a mean of the wrong shape raises an error with an appropriate message."""
        batch_shape = torch.Size([NUM_LATENTS + 2])
        with self.assertRaises(ValueError) as ctx:
            SoftmaxClassifier(
                self.dataset.train_x,
                self.dataset.train_y,
                kernel_class=ScaledRBFKernel,
                mean_class=BatchScaledMean,
                y_std=0,
                likelihood_class=SoftmaxLikelihood,
                marginal_log_likelihood_class=VariationalELBO,
                mean_kwargs={"batch_shape": batch_shape},
                rng=self.rng,
            )

        self.assertEqual(
            f"Expected a batch shape of torch.Size([{NUM_LATENTS}]); got {batch_shape!r}.",
            str(ctx.exception),
        )


class MultitaskBernoulliClassifierTests(ClassificationTestCase):
    """
    Tests for softmax multi-class classification with LMC.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.rng = get_default_rng()
        self.dataset = MulticlassGaussianClassificationDataset(
            num_train_points=60, num_test_points=20, num_classes=4, rng=self.rng
        )

        self.controller = MultitaskBernoulliClassifier(
            self.dataset.train_x,
            one_hot(self.dataset.train_y),
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=MultitaskBernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            rng=self.rng,
        )

    def test_predictions(self) -> None:
        """Predict on a test dataset, and check the predictions are reasonably accurate."""
        self.controller.fit(10)
        predictions, _ = self.controller.classify_points(self.dataset.test_x)
        self.assertPredictionsEqual(self.dataset.test_y, predictions, delta=0.4)
