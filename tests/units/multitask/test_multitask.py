"""
Tests for the Multitask decorator.
"""

import unittest
from unittest.mock import MagicMock

import gpytorch
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational.variational_strategy import VariationalStrategy

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.models import ExactGPModel
from vanguard.multitask import Multitask
from vanguard.multitask.decorator import _multitaskify_mean
from vanguard.multitask.kernel import BatchCompatibleMultitaskKernel
from vanguard.multitask.models import independent_variational_multitask_model, multitask_model
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference


# pylint: disable=abstract-method
class ApproxGPModel(ApproximateGP):
    """
    Approximate GP model to use for testing.
    """

    def __init__(
        self,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        variational_strategy: VariationalStrategy,
    ) -> None:
        """Initialise self."""
        super().__init__(variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module


# pylint: enable=abstract-method


class ErrorTests(unittest.TestCase):
    """
    Tests that the correct error messages are thrown.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.rng = get_default_rng()
        self.dataset = SyntheticDataset(rng=self.rng)

    def test_single_task_variational(self) -> None:
        """Test that variational inference with a single task throws an error."""

        @Multitask(num_tasks=1)
        @VariationalInference()
        class MultitaskController(GaussianGPController):
            pass

        with self.assertRaisesRegex(TypeError, "You are using a multitask variational model in a single-task problem."):
            MultitaskController(
                self.dataset.train_x, self.dataset.train_y, ScaledRBFKernel, self.dataset.train_y_std, rng=self.rng
            )

    def test_bad_batch_shape(self) -> None:
        """
        Test how the multitask decorator handles an invalid batch shape.

        The batch shape passed here is not an iterable, but will internally be passed after a * line in a method input,
        suggesting it should be an iterable.
        """

        @Multitask(num_tasks=1)
        class MultitaskController(GaussianGPController):
            pass

        with self.assertRaisesRegex(TypeError, "must be an iterable, not int"):
            MultitaskController(
                self.dataset.train_x,
                self.dataset.train_y,
                ScaledRBFKernel,
                self.dataset.train_y_std,
                kernel_kwargs={"batch_shape": 2},
                rng=self.rng,
            )


class TestMulticlassModels(unittest.TestCase):
    """
    Tests for constructing multiclass models.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.train_x = torch.tensor([1, 2, 3])
        self.train_y = torch.tensor([[4, 5], [6, 7], [8, 9]])
        self.num_tasks = 2
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = BatchCompatibleMultitaskKernel(
            data_covar_module=ScaledRBFKernel(),
            num_tasks=2,
        )
        self.rng = get_default_rng()

    def test_multitask_model_exact_gp_model(self) -> None:
        """Test construction of a multitask model when using an exact GP model."""

        @multitask_model
        class MultitaskModel(ExactGPModel):
            pass

        # Create the model
        model = MultitaskModel(
            train_x=self.train_x,
            train_y=self.train_y,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            mean_module=gpytorch.means.MultitaskMean(num_tasks=self.num_tasks, base_means=self.mean_module),
            covar_module=self.covar_module,
        )

        # Pass data forward, and verify the type and shape of the output
        result = model(self.train_x)

        # We expect a MultitaskMultivariateNormal distribution out - since we are using a multitask model and
        # an exact GP
        self.assertIsInstance(result, MultitaskMultivariateNormal)
        self.assertListEqual(list(result.mean.shape), [self.train_x.shape[0], self.train_y.shape[1]])

    def test_multitask_model_approx_gp_model(self) -> None:
        """Test construction of a multitask model when using an approximate GP model."""

        # pylint: disable=abstract-method
        @multitask_model
        class MultitaskModel(ApproxGPModel):
            pass

        # pylint: enable=abstract-method
        # Create the model
        model = MultitaskModel(
            mean_module=gpytorch.means.MultitaskMean(num_tasks=self.num_tasks, base_means=self.mean_module),
            covar_module=self.covar_module,
            variational_strategy=MagicMock(),
        )

        # Pass data forward, and verify the type and shape of the output
        result = model.forward(self.train_x)

        # We expect a MultivariateNormal distribution out - since we are using a multitask model and
        # an approximate GP
        self.assertIsInstance(result, MultivariateNormal)
        self.assertListEqual(list(result.mean.shape), [self.train_x.shape[0], self.train_y.shape[1]])

    def test_multitask_model_invalid_gp_model(self) -> None:
        """Test construction of a multitask model when using an invalid GP model."""
        # Create the model - should raise a type error since it's not a child of an exact or approximate GP
        with self.assertRaisesRegex(TypeError, "Must be applied to a subclass of 'ExactGP' or 'ApproximateGP'."):
            # pylint: disable=unused-variable
            @multitask_model
            class MultitaskModel(MagicMock):
                pass

            # pylint: enable=unused-variable

    def test_independent_variational_multitask_model(self) -> None:
        """Test decoration using independent_variational_multitask_model with a valid and invalid mean module."""

        # pylint: disable=abstract-method
        @independent_variational_multitask_model
        class MultitaskModel(ApproxGPModel):
            pass

        # pylint: enable=abstract-method

        # Setup a mean module just for this test
        mean_module = gpytorch.means.ConstantMean()
        mean_module.batch_shape = [1, 2, 3]

        # Get the number of latent variables for a valid and invalid mean module. In the valid case, we expect
        # the last element of the batch shape on the mean module
        # pylint: disable=protected-access
        self.assertEqual(MultitaskModel._get_num_latents(mean_module), mean_module.batch_shape[-1])
        # pylint: enable=protected-access

        # If batch shape is just an integer, we should get a type error when trying to index the integer
        mean_module.batch_shape = 3
        with self.assertRaisesRegex(
            TypeError, "'mean_module.batch_shape' must be subscriptable, cannot index given value."
        ):
            # pylint: disable=protected-access
            MultitaskModel._get_num_latents(mean_module)
            # pylint: enable=protected-access

        # if batch shape is empty, we should get an index error, but this should then be transformed into a type
        # error inside the code detailing to the user the issue
        mean_module.batch_shape = []
        with self.assertRaisesRegex(TypeError, "a one-dimensional, non-zero length batch shape is required"):
            # pylint: disable=protected-access
            MultitaskModel._get_num_latents(mean_module)
            # pylint: enable=protected-access


class TestMulticlassMeans(unittest.TestCase):
    """
    Test conversion of mean objects.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.num_tasks = 2

    def test_conversion(self) -> None:
        """Test conversion of different mean objects to multitask means objects."""
        # If we try to convert an already multitask mean, we should just get out what we put in
        self.assertIsInstance(
            _multitaskify_mean(gpytorch.means.MultitaskMean, self.num_tasks)(
                base_means=[gpytorch.means.ConstantMean()], num_tasks=self.num_tasks
            ),
            gpytorch.means.MultitaskMean,
        )

        # If we try to convert a non-multitask mean, we should not get out what we put in. Note that we start with a
        # ConstantMean object, but when we initialise the output of _multitaskify_kernel, it's now of a different type
        self.assertIsInstance(
            _multitaskify_mean(gpytorch.means.ConstantMean, self.num_tasks)(), gpytorch.means.MultitaskMean
        )


class TestMulticlassDecorator(unittest.TestCase):
    """
    Test Multiclass decorator.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.train_x = torch.tensor([1, 2, 3])
        self.train_y = torch.tensor([[4, 5], [6, 7], [8, 9]])
        self.rng = get_default_rng()

    def test_likelihood_noise(self) -> None:
        """Test that the likelihood_noise property is handled as expected."""

        @Multitask(num_tasks=2)
        @VariationalInference()
        class MultitaskController(GaussianGPController):
            pass

        # Create the controller
        gp = MultitaskController(
            train_x=self.train_x,
            train_y=self.train_y,
            kernel_class=ScaledRBFKernel,
            y_std=0.0,
            likelihood_class=gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            rng=self.rng,
        )

        # Try and get the noise - if we've not set it yet, we should get an attribute error
        with self.assertRaisesRegex(
            AttributeError,
            "'fixed_noise' appears to have not been set yet. This can be set with the `likelihood_noise` method",
        ):
            _ = gp.likelihood_noise

        # Set the noise, then ensure it's been set as expected
        noise = torch.tensor([1989.0, 22.0, 15.0])
        gp.likelihood_noise = noise

        # The noise should now be set as above, but it should have been set onto the likelihood
        # object internal to the GP
        torch.testing.assert_close(gp.likelihood_noise, noise)
        torch.testing.assert_close(gp.likelihood.fixed_noise, noise)
