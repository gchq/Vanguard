"""
Tests for models.
"""

import unittest
from unittest.mock import MagicMock, patch

import gpytorch
import torch

from vanguard.kernels import ScaledRBFKernel
from vanguard.models import ExactGPModel, InducingPointKernelGPModel


class TestExactGPModel(unittest.TestCase):
    """
    Tests for ExactGPModel.
    """

    def test_forward(self) -> None:
        """Test creation and forward pass using the model."""
        # Setup for the model
        example_x = torch.tensor([0.0, 1.0, 2.0])
        example_y = torch.tensor([5.0, 6.0, 7.0])

        # Create the model
        kernel = ScaledRBFKernel()
        model = ExactGPModel(
            train_x=example_x,
            train_y=example_y,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            mean_module=gpytorch.means.ConstantMean(),
            covar_module=kernel,
        )

        # Set expected outputs - the constant mean used should result in a mean vector
        # of zeros
        expected_mean = torch.tensor([0.0, 0.0, 0.0])

        # Set expected outputs - the covariance matrix should be the kernel evaluated on
        # the provided data
        expected_covariance = kernel(example_x, torch.transpose(example_x, 0, 0)).to_dense()

        # Compute a forward pass on the data
        result = model.forward(torch.tensor([0.0, 1.0, 2.0]))

        # Check outputs match expected
        torch.testing.assert_close(expected_mean, result.loc)
        torch.testing.assert_close(expected_covariance, result.covariance_matrix)


class TestInducingPointKernelGPModel(unittest.TestCase):
    """
    Tests for InducingPointKernelGPModel.
    """

    def test_creation(self) -> None:
        """Test creation of the model."""
        # Setup for the model
        example_x = torch.tensor([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0], [50.0, 500.0]])
        example_y = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0])
        n_inducing_points = 3

        # Create the model
        kernel = ScaledRBFKernel()

        with patch("gpytorch.kernels.InducingPointKernel") as patched_kernel:
            # Mock the random generation to pick some fixed indices
            mocked_rng = MagicMock()
            mocked_choice = MagicMock()
            mocked_choice.return_value = [1, 2, 4]
            mocked_rng.choice = mocked_choice

            # Create the model, which will have our mocked and patch objects manipulate the
            # creation internally
            InducingPointKernelGPModel(
                train_x=example_x,
                train_y=example_y,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                mean_module=gpytorch.means.ConstantMean(),
                covar_module=kernel,
                n_inducing_points=n_inducing_points,
                rng=mocked_rng,
            )

            # Check the kernel was created with the pre-selected points (done via mocking) by
            # inspecting the call to the patched kernel class. We inspect if the expected inducing
            # points were actually passed
            self.assertEqual(patched_kernel.call_count, 1)
            torch.testing.assert_close(patched_kernel.call_args_list[0][1]["inducing_points"], example_x[[1, 2, 4], :])

            # Check the calls to the patched random choice - arguments should be the number of
            # points in example_x, the number of inducing points desired and sampling with replacement
            mocked_choice.assert_called_once_with(example_x.shape[0], size=n_inducing_points, replace=True)
