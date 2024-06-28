"""Tests for DirichletKernelClassifierLikelihood."""

from unittest import TestCase

import numpy as np
import torch.testing
from gpytorch import lazify
from gpytorch.kernels import RBFKernel
from gpytorch.means import ZeroMean

from vanguard.classification.kernel import DirichletKernelMulticlassClassification
from vanguard.classification.likelihoods import DirichletKernelClassifierLikelihood, GenericExactMarginalLogLikelihood
from vanguard.classification.models import DummyKernelDistribution
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.vanilla import GaussianGPController

NUM_CLASSES = 4


@DirichletKernelMulticlassClassification(num_classes=NUM_CLASSES, ignore_methods=("__init__",))
class MulticlassGaussianClassifier(GaussianGPController):
    """A simple Dirichlet multiclass classifier."""


class TestDirichletKernelClassifierLikelihood(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = MulticlassGaussianClassificationDataset(
            num_train_points=NUM_CLASSES * 3, num_test_points=NUM_CLASSES, num_classes=NUM_CLASSES, seed=1234
        )

    def setUp(self):
        self.rng = np.random.default_rng(1234)

    def test_illegal_input_type(self):
        """Test that we get an appropriate error when an illegal argument type is passed."""
        likelihood = DirichletKernelClassifierLikelihood(num_classes=NUM_CLASSES)

        # various illegal inputs
        illegal_inputs = [object(), np.array([1, 2, 3]), "string"]

        for illegal_input in illegal_inputs:
            with self.subTest(repr(illegal_input)):
                with self.assertRaises(TypeError) as ctx:
                    # ignore type: it's intentionally incorrect
                    likelihood(illegal_input)  # type: ignore
                self.assertEqual(
                    "Likelihoods expects a DummyKernelDistribution input to make marginal predictions, or a "
                    f"torch.Tensor for conditional predictions. Got a {type(illegal_input).__name__}",
                    ctx.exception.args[0],
                )

    def test_alpha(self):
        """Test that when a value for alpha is provided, it's set correctly."""
        alpha = self.rng.uniform(2, 10)  # ensuring alpha != 1
        likelihood = DirichletKernelClassifierLikelihood(num_classes=NUM_CLASSES, alpha=alpha)
        torch.testing.assert_close(torch.ones(NUM_CLASSES) * alpha, likelihood.alpha)

    def test_learn_alpha(self):
        """Test that when learn_alpha is True, its value is changed during fitting."""
        # TODO: also test alpha_prior and alpha_constraint?

        controller = MulticlassGaussianClassifier(
            train_x=self.dataset.train_x,
            train_y=self.dataset.train_y,
            y_std=0,
            mean_class=ZeroMean,
            kernel_class=RBFKernel,
            likelihood_class=DirichletKernelClassifierLikelihood,
            likelihood_kwargs={"learn_alpha": True, "alpha": 10},
            marginal_log_likelihood_class=GenericExactMarginalLogLikelihood,
        )

        starting_alpha = controller.likelihood.alpha.clone()
        controller.fit(1)
        fitted_alpha = controller.likelihood.alpha

        # assert that alpha has changed
        assert not torch.all(torch.isclose(fitted_alpha, starting_alpha))

    def test_log_marginal(self):
        """Test that log_marginal gives the log-probabilities of the marginal distribution."""
        likelihood = DirichletKernelClassifierLikelihood(num_classes=NUM_CLASSES)
        kernel = torch.tensor(self.rng.uniform(1, 2, size=(NUM_CLASSES, NUM_CLASSES)), dtype=torch.float)
        distribution = DummyKernelDistribution(lazify(torch.eye(NUM_CLASSES, dtype=torch.float)), lazify(kernel))
        observations = torch.tensor(self.rng.standard_normal(size=NUM_CLASSES), dtype=torch.float)

        log_prob_direct = likelihood.log_marginal(observations, distribution)
        log_prob_indirect = likelihood.marginal(distribution).log_prob(observations)
        torch.testing.assert_close(log_prob_direct, log_prob_indirect)
