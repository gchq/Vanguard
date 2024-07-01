"""Tests for the InertKernelModel."""

from unittest import TestCase, expectedFailure

import torch
from gpytorch import settings
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.utils.warnings import GPInputWarning

from vanguard.classification.models import InertKernelModel


class TestInertKernelModelFailures(TestCase):
    def test_train_fails_with_no_data(self):
        """Test that model training fails with an appropriate error message if no training inputs are provided."""
        model = InertKernelModel(
            train_inputs=None,
            train_targets=None,
            covar_module=RBFKernel(),
            mean_module=None,
            likelihood=BernoulliLikelihood(),
            num_classes=3,
        )

        with self.assertRaises(RuntimeError) as ctx:
            model.train()

        self.assertEqual(
            "train_inputs, train_targets cannot be None in training mode. "
            "Call .eval() for prior predictions, or call .set_train_data() to add training data.",
            ctx.exception.args[0],
        )

    def test_illegal_train_inputs(self):
        """Test that model training fails with an appropriate message if training inputs are of an incorrect type."""
        with self.assertRaises(TypeError) as ctx:
            _ = InertKernelModel(
                train_inputs=[1, 2, 3],  # type: ignore
                train_targets=None,
                covar_module=RBFKernel(),
                mean_module=None,
                likelihood=BernoulliLikelihood(),
                num_classes=3,
            )

        self.assertEqual("Train inputs must be a tensor, or a list/tuple of tensors", ctx.exception.args[0])


class TestInertKernelModel(TestCase):
    def setUp(self):
        # Simple three-class training data.
        self.train_data = torch.tensor([0.0, 0.1, 0.4, 0.5, 0.9, 1.0])
        self.train_targets = torch.tensor([0, 0, 1, 1, 2, 2])

        # ... and some accompanying test data.
        self.test_data = torch.tensor([0.05, 0.89])

        self.model = InertKernelModel(
            train_inputs=self.train_data,
            train_targets=self.train_targets,
            covar_module=RBFKernel(),
            mean_module=None,
            likelihood=BernoulliLikelihood(),
            num_classes=3,
        )

    def test_train_fails_in_debug_if_input_is_not_train_input(self):
        """Test that, when in debug mode, training fails if inputs other than the training inputs are provided."""
        other_input = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        with self.assertRaises(RuntimeError) as ctx:
            with settings.debug(True):
                self.model.train()
                self.model(other_input)

        self.assertEqual("You must train on the training inputs!", ctx.exception.args[0])

    def test_using_training_data_outside_train_mode_warns_in_debug_if_forget_to_call_train(self):
        """Test that, when in debug mode, calling the model with the training data emits an appropriate warning."""
        with self.assertWarns(
            GPInputWarning, msg="The input matches the stored training data. Did you forget to call model.train()?"
        ):
            with settings.debug(True):
                self.model.eval()
                self.model(self.train_data)

    # TODO: all three prior mode tests currently fail due to shape mismatches.
    # https://github.com/gchq/Vanguard/issues/291
    @expectedFailure
    def test_prior_mode(self):
        """Test that when in prior mode, the GP is evaluated as if without training data."""
        # Predict in prior mode
        self.model.eval()
        with settings.prior_mode(True):
            prior_mode_distribution = self.model(self.test_data)

        # Assert the distribution uses the prior distribution and not the training data
        torch.testing.assert_close(
            prior_mode_distribution.kernel.to_dense().T, self.model.covar_module(self.test_data).to_dense()
        )

    @expectedFailure
    def test_prior_mode_if_no_train_inputs(self):
        """Test that if no training inputs are present, the model is evaluated as if in prior mode."""
        # Predict in eval mode
        self.model.train_inputs = None
        self.model.eval()
        prior_mode_distribution = self.model(self.test_data)

        # Assert the distribution uses the prior distribution and not the training data
        torch.testing.assert_close(
            prior_mode_distribution.kernel.to_dense().T, self.model.covar_module(self.test_data).to_dense()
        )

    @expectedFailure
    def test_prior_mode_if_no_train_targets(self):
        """Test that if no training targets are present, the model is evaluated as if in prior mode."""
        # Predict in eval mode
        self.model.train_targets = None
        self.model.eval()
        prior_mode_distribution = self.model(self.test_data)

        # Assert the distribution uses the prior distribution and not the training data
        torch.testing.assert_close(
            prior_mode_distribution.kernel.to_dense().T, self.model.covar_module(self.test_data).to_dense()
        )

    def test_eval_mode(self):
        """Test that in eval mode, the GP is evaluated with the training data."""
        # Train the model
        self.model.train()
        self.model(self.train_data)

        # Predict in eval mode
        self.model.eval()
        distribution = self.model(self.test_data)

        # The distribution kernel should be the covariance between the training data and the test data
        torch.testing.assert_close(
            distribution.kernel.to_dense().T, self.model.covar_module(self.train_data, self.test_data).to_dense()
        )
