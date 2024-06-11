"""Tests for the InertKernelModel."""

from unittest import TestCase

import torch
from gpytorch import settings
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.utils.warnings import GPInputWarning

from vanguard.classification.models import InertKernelModel


class TestInertKernelModelFailures(TestCase):
    def test_train_fails_with_no_data(self):
        """Test that model training fails if no training inputs are provided."""
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
        """Test that model training fails if training inputs are of an illegal type."""
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
        """Test that, when in debug mode, calling the model with the training data raises an appropriate warning."""
        with self.assertWarns(
            GPInputWarning, msg="The input matches the stored training data. Did you forget to call model.train()?"
        ):
            with settings.debug(True):
                self.model.train(False)
                self.model(self.train_data)
