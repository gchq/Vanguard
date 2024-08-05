"""
Tests for the Multitask likelihoods.
"""

import unittest

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from linear_operator.operators import DiagLinearOperator

from vanguard.multitask.likelihoods import FixedNoiseMultitaskGaussianLikelihood


class LikelihoodTests(unittest.TestCase):
    """
    Tests functionality of multitask likelihoods.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.noise_tensor = torch.tensor([1.0, 2.0, 3.0])
        self.default_batch_shape = torch.Size([2, 2])
        self.num_tasks = 2
        self.model = FixedNoiseMultitaskGaussianLikelihood(
            noise=self.noise_tensor,
            learn_additional_noise=False,
            batch_shape=self.default_batch_shape,
            num_tasks=self.num_tasks,
        )

    def test_fixed_noise(self) -> None:
        """Test that the fixed noise attribute can be accessed and set as expected."""
        # The value of fixed_noise should have been set to the noise tensor we passed when creating
        # the object
        torch.testing.assert_close(self.model.fixed_noise, self.noise_tensor)

    def test_setting_fixed_noise(self) -> None:
        """Test that the fixed noise attribute can be set manually."""
        model = FixedNoiseMultitaskGaussianLikelihood(
            noise=self.noise_tensor,
            learn_additional_noise=False,
            batch_shape=self.default_batch_shape,
            num_tasks=self.num_tasks,
        )
        model.fixed_noise = torch.tensor([0.0])
        torch.testing.assert_close(self.model.fixed_noise, self.noise_tensor)

    def test_marginal(self) -> None:
        """Test creation of a marginal distribution using a likelihood object."""
        # Define some test specific data
        mean = torch.tensor([[5.0, 6.0, 7.0]])
        covariance_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        function_dist = MultitaskMultivariateNormal(mean, covariance_matrix)
        marginal = self.model.marginal(function_dist)

        # We should have the marginal distribution as a MultitaskMultivariateNormal object, with
        # the mean as passed on creation and the covariance matrix as the passed value plus the noise
        # on the original modelling object
        self.assertIsInstance(marginal, MultitaskMultivariateNormal)
        torch.testing.assert_close(marginal.mean, mean)
        torch.testing.assert_close(marginal.covariance_matrix, covariance_matrix + torch.diag(self.noise_tensor))

    def test_flatten_noise(self) -> None:
        """Test the method _flatten_noise functions as expected"""
        # Define an example input and expected output - which should be the provided noise, but reshaped into a 1
        # dimensional array
        noise_to_flatten = torch.tensor([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
        expected_output = torch.tensor([0.1, 1.0, 0.2, 2.0, 0.3, 3.0])

        # Check output matches expected
        # pylint: disable-next=protected-access
        torch.testing.assert_close(self.model._flatten_noise(noise_to_flatten), expected_output)

    def test_shaped_noise_covar(self) -> None:
        """Test the method _shaped_noise_covar functions as expected."""
        # pylint: disable-next=protected-access
        result = self.model._shaped_noise_covar(base_shape=self.default_batch_shape)
        self.assertIsInstance(result, DiagLinearOperator)
        torch.testing.assert_close(result.diagonal(), self.noise_tensor)

        # We have specified that additional noise should not be learnt, so we do not expect the
        #  resulting noise diagonal to require gradients
        self.assertFalse(result.diagonal().requires_grad)

    def test_shaped_noise_covar_learn_additional_noise(self) -> None:
        """Test the method _shaped_noise_covar functions as expected when learning additional noise."""
        # Setup a minimal example for the test
        model = FixedNoiseMultitaskGaussianLikelihood(
            noise=torch.tensor([0.0]),
            learn_additional_noise=True,
            batch_shape=torch.Size([]),
            num_tasks=self.num_tasks,
        )
        # pylint: disable-next=protected-access
        result = model._shaped_noise_covar(base_shape=self.default_batch_shape)

        # Check the type of the output is as expected
        self.assertIsInstance(result, DiagLinearOperator)

        # We have specified that additional noise should be learnt, so we expect the resulting noise diagonal
        # to require gradients
        self.assertTrue(result.diagonal().requires_grad)
