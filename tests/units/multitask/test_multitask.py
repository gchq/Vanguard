# © Crown Copyright GCHQ
#
# Licensed under the GNU General Public License, version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for the Multitask decorator.
"""

import unittest
from typing import Callable, Optional
from unittest.mock import MagicMock, Mock, patch

import gpytorch
import pytest
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational.variational_strategy import VariationalStrategy

from tests.cases import get_default_rng, maybe_throws
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.models import ExactGPModel
from vanguard.multitask import Multitask
from vanguard.multitask.decorator import _multitaskify_mean
from vanguard.multitask.kernel import BatchCompatibleMultitaskKernel
from vanguard.multitask.models import (
    independent_variational_multitask_model,
    lmc_variational_multitask_model,
    multitask_model,
)
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


@Multitask(num_tasks=2)
@VariationalInference()
class VariationalInferenceMultitaskController(GaussianGPController):
    pass


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

    @pytest.mark.no_beartype
    def test_bad_batch_shape_on_kernel(self) -> None:
        """
        Test how the multitask decorator handles an invalid batch shape in the kernel keyword arguments.

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


@pytest.mark.parametrize(
    ["model_decorator", "expected_exc_type", "expected_exc_message"],
    [
        (
            independent_variational_multitask_model,
            ValueError,
            "You are using a multitask variational model which requires that `num_tasks==num_latents`",
        ),
        (lmc_variational_multitask_model, None, None),
    ],
)
def test_variational_multitask_model_task_latent_mismatch(
    model_decorator: Callable[[type], type],
    expected_exc_type: Optional[type[Exception]],
    expected_exc_message: Optional[str],
) -> None:
    """
    Test what happens when the number of latent dims and tasks do not agree.

    For an `independent_variational_multitask_model`, an error should be thrown, as this is invalid.

    For an `lmc_variational_multitask_model`, no error should be thrown.
    """

    # pylint: disable=abstract-method
    @model_decorator
    class MultitaskModel(ApproxGPModel):
        pass

    class MockMultitaskModel(MultitaskModel):
        # pylint: disable-next=super-init-not-called
        def __init__(self):
            # We explicitly *don't* call super().__init__() here, so we skip all the checks that would otherwise be
            # there and can look only at the one check being tested.
            self.num_tasks = 2
            self.num_latents = 3
            # We define the number of tasks and the number of latents to be different.

    # pylint: enable=abstract-method

    # Minimal example to only define the data necessary for this test
    mock_model = MockMultitaskModel()
    mean_module = gpytorch.means.ConstantMean()
    covar_module = BatchCompatibleMultitaskKernel(data_covar_module=ScaledRBFKernel(), num_tasks=2)
    covar_module.batch_shape = [5, 3]

    # Note above that mock_model.num_tasks does not equal mock_model.num_latents, which is invalid for an independent
    # variational model, but valid for an LMC model. The user should be informed via a relevant error if it's
    # invalid, and if it's valid no error should be thrown.
    with maybe_throws(expected_exc_type, match=expected_exc_message):
        # pylint: disable-next=protected-access
        mock_model._check_batch_shape(mean_module=mean_module, covar_module=covar_module)


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

    @pytest.mark.no_beartype
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
        # pylint: disable-next=protected-access
        self.assertEqual(MultitaskModel._get_num_latents(mean_module), mean_module.batch_shape[-1])

        # If batch shape is just an integer, we should get a type error when trying to index the integer
        mean_module.batch_shape = 3
        with self.assertRaisesRegex(
            TypeError, "'mean_module.batch_shape' must be subscriptable, cannot index given value."
        ):
            # pylint: disable-next=protected-access
            MultitaskModel._get_num_latents(mean_module)

        # If batch_shape is empty, we should get an index error, but this should then be transformed into a type
        # error inside the code detailing to the user the issue
        mean_module.batch_shape = []
        with self.assertRaisesRegex(TypeError, "a one-dimensional, non-zero length batch shape is required"):
            # pylint: disable-next=protected-access
            MultitaskModel._get_num_latents(mean_module)


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
        self.train_x = torch.tensor([1.0, 2.0, 3.0])
        self.train_y = torch.tensor([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
        self.num_tasks = 2
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = BatchCompatibleMultitaskKernel(
            data_covar_module=ScaledRBFKernel(),
            num_tasks=2,
        )
        self.rng = get_default_rng()

    def test_likelihood_noise(self) -> None:
        """Test that the likelihood_noise property is handled as expected."""
        # Create the controller
        gp = VariationalInferenceMultitaskController(
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

    def test_gp_model_class(self) -> None:
        """Test construction of a multitask model when using a linear model of co-regionalisation."""

        @Multitask(num_tasks=2, lmc_dimension=2)
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

        # Update mean and covariance module to test the generated GP model class
        temp_mean = self.mean_module
        temp_covar = self.covar_module
        temp_mean.batch_shape = [2]
        temp_covar.batch_shape = [2]
        sub_gp = gp.gp_model_class(
            train_x=self.train_x,
            train_y=self.train_y,
            likelihood=gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.zeros_like(self.train_x)),
            mean_module=temp_mean,
            covar_module=self.covar_module,
            n_inducing_points=1,
            num_tasks=self.num_tasks,
            rng=self.rng,
        )

        # Since we have used lmc_dimension, we should have internally set gp_model_class to use
        # an LMCVariationalStrategy - check this is indeed true
        self.assertIsInstance(sub_gp.variational_strategy, gpytorch.variational.LMCVariationalStrategy)

    def test_invalid_batch_shape_type(self) -> None:
        """
        Test construction of a multitask model when passing an invalid batch shape (wrong type) as a keyword.

        If we pass batch shape as a keyword argument, it must be as a torch.Size() object, and match the batch shape
        for the kernel. If not, it should raise a relevant error.
        """
        with self.assertRaisesRegex(
            TypeError, r"Expected mean_kwargs\['batch_shape'\] to be of type `torch.Size`; got `list` instead"
        ):
            VariationalInferenceMultitaskController(
                train_x=self.train_x,
                train_y=self.train_y,
                kernel_class=ScaledRBFKernel,
                y_std=0.0,
                likelihood_class=gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
                marginal_log_likelihood_class=VariationalELBO,
                mean_kwargs={"batch_shape": [22]},
                rng=self.rng,
            )

    def test_unexpected_match_mean_shape_type_error_reraised(self) -> None:
        """
        Test that if `_match_mean_shape_to_kernel` raises an unexpected `TypeError`, it is reraised.

        See #357 - previously, this would have been silently suppressed within `__init__`.
        """
        with (
            self.assertRaisesRegex(TypeError, "Testing error"),
            patch.object(
                VariationalInferenceMultitaskController,
                "_match_mean_shape_to_kernel",
                Mock(side_effect=TypeError("Testing error")),
            ),
        ):
            VariationalInferenceMultitaskController(
                train_x=self.train_x,
                train_y=self.train_y,
                kernel_class=ScaledRBFKernel,
                y_std=0.0,
                likelihood_class=gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
                marginal_log_likelihood_class=VariationalELBO,
                mean_kwargs={"batch_shape": torch.Size([22])},
                kernel_kwargs={"batch_shape": torch.Size([22])},
                rng=self.rng,
            )

    def test_valid_batch_shape(self) -> None:
        """
        Test construction of a multitask model when passing a valid batch shape as a keyword.

        The batch shape must be a :py:class:`torch.Size` instance, and must match the batch shape for the kernel.
        """
        # As a sense check, verify that if we pass an expected batch shape we do not get an error and it
        # is set as expected
        gp = VariationalInferenceMultitaskController(
            train_x=self.train_x,
            train_y=self.train_y,
            kernel_class=ScaledRBFKernel,
            y_std=0.0,
            likelihood_class=gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
            mean_kwargs={"batch_shape": torch.Size([22])},
            kernel_kwargs={"batch_shape": torch.Size([22])},
            rng=self.rng,
        )
        self.assertListEqual(list(gp.mean.batch_shape), [22, self.train_y.shape[1]])

    def test_match_mean_shape_to_multitask_kernel(self) -> None:
        """Test usage of _match_mean_shape_to_kernel when provided a multitask kernel."""
        # Call the method, which should output an uninstantiated multiclass mean object, which we
        # then instantiate and verify is in fact a multiclass mean object.
        # pylint: disable-next=protected-access, no-member
        result = VariationalInferenceMultitaskController._match_mean_shape_to_kernel(
            mean_class=gpytorch.means.ConstantMean,
            kernel_class=BatchCompatibleMultitaskKernel,
            mean_kwargs={},
            kernel_kwargs={"data_covar_module": ScaledRBFKernel(), "num_tasks": self.num_tasks},
        )()

        # We should have a multitask mean object, and we should have the correct number of
        # tasks represented
        self.assertIsInstance(result, gpytorch.means.MultitaskMean)
        self.assertEqual(len(result.base_means), self.num_tasks)

    def test_match_mean_shape_to_kernel_invalid(self) -> None:
        """
        Test usage of _match_mean_shape_to_kernel when provided an invalid input setup.

        In this test, we specify a different batch size for the mean and the kernel - which does
        not make sense and should raise an error informing the user.
        """
        with self.assertRaisesRegex(
            ValueError,
            r"The provided mean has batch_shape torch.Size\(\[3, 4\]\) but the provided kernel "
            r"has batch_shape torch.Size\(\[2, 3\]\). They must match.",
        ):
            # pylint: disable-next=protected-access, no-member
            VariationalInferenceMultitaskController._match_mean_shape_to_kernel(
                mean_class=gpytorch.means.ConstantMean,
                kernel_class=ScaledRBFKernel,
                mean_kwargs={"batch_shape": torch.Size([3, 4])},
                kernel_kwargs={"batch_shape": torch.Size([2, 3])},
            )
