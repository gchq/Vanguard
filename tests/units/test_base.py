"""
Tests for the GPController class.
"""

import unittest
from typing import Union

# the numpy.typing import _is_ used, but only as `np.typing`, so this is a false positive from pylint
import gpytorch
import numpy as np
import numpy.typing  # pylint: disable=unused-import
import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from vanguard.base import GPController
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import PeriodicRBFKernel, ScaledRBFKernel
from vanguard.optimise import SmartOptimiser
from vanguard.vanilla import GaussianGPController

from ..cases import VanguardTestCase


class DefaultTensorTypeTests(unittest.TestCase):
    """
    Tests for the setting of the default tensor types.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.original_default_tensor_type = GaussianGPController.get_default_tensor_type()
        self.original_dtype = self.original_default_tensor_type.dtype
        self.original_is_cuda = self.original_default_tensor_type.is_cuda

        if self.original_default_tensor_type == torch.DoubleTensor:
            self.skipTest("This test us voided because the default tensor type would not be changed.")

        original_tensor = torch.tensor([])
        self.assertEqual(original_tensor.dtype, self.original_dtype)
        self.assertEqual(original_tensor.is_cuda, self.original_is_cuda)

        class NewController(GaussianGPController):
            pass

        self.new_controller_class = NewController
        self.new_controller_class.set_default_tensor_type(torch.DoubleTensor)

    def tearDown(self) -> None:
        """Code to run after each test."""
        self.new_controller_class.set_default_tensor_type(self.original_default_tensor_type)
        tensor = torch.tensor([])
        self.assertEqual(tensor.dtype, self.original_dtype)
        self.assertEqual(tensor.is_cuda, self.original_is_cuda)

    def test_class_default_tensor(self) -> None:
        """Should have changed."""
        self.assertEqual(self.new_controller_class.get_default_tensor_type(), torch.DoubleTensor)

    def test_superclass_default_tensor(self) -> None:
        """Should be unchanged."""
        self.assertEqual(GaussianGPController.get_default_tensor_type(), self.original_default_tensor_type)

    def test_default_tensor(self) -> None:
        """New tensors should now match."""
        new_tensor = torch.tensor([])
        self.assertEqual(new_tensor.dtype, torch.float64)
        self.assertEqual(new_tensor.is_cuda, False)


class InputTests(VanguardTestCase):
    """
    GP controllers are forgiving about the shape of data arrays, where possible.
    These tests check that this behaviour.
    """

    DATASET = SyntheticDataset()

    def test_unsqueeze_y(self) -> None:
        """Make sure the tensor wrangling works if we pass y with the wrong shape."""
        squeezed_train_y = self.DATASET.train_y.reshape(-1, 1)
        gp = GPController(
            train_x=self.DATASET.train_x,
            train_y=squeezed_train_y,
            kernel_class=PeriodicRBFKernel,
            mean_class=ConstantMean,
            y_std=self.DATASET.train_y_std,
            likelihood_class=FixedNoiseGaussianLikelihood,
            marginal_log_likelihood_class=ExactMarginalLogLikelihood,
            optimiser_class=torch.optim.Adam,
            smart_optimiser_class=SmartOptimiser,
        )
        np.testing.assert_array_almost_equal(squeezed_train_y, gp.train_y.detach().cpu().numpy(), decimal=5)

    def test_unsqueeze_x(self) -> None:
        """Check the tensor wrangling works if we pass x with the wrong shape."""
        train_x_mean = self.DATASET.train_x.ravel()
        gp = GPController(
            train_x=train_x_mean,
            train_y=self.DATASET.train_y,
            kernel_class=PeriodicRBFKernel,
            mean_class=ConstantMean,
            y_std=self.DATASET.train_y_std,
            likelihood_class=FixedNoiseGaussianLikelihood,
            marginal_log_likelihood_class=ExactMarginalLogLikelihood,
            optimiser_class=torch.optim.Adam,
            smart_optimiser_class=SmartOptimiser,
        )
        np.testing.assert_array_almost_equal(self.DATASET.train_x, gp.train_x.detach().cpu().numpy(), decimal=5)

    def test_error_handling_of_higher_rank_features(self) -> None:
        """Check that shape errors due to incorrectly treated high-rank features are caught and explained."""
        shape = (len(self.DATASET.train_y), 31, 4)
        train_x_mean = np.random.randn(*shape)
        gp = GaussianGPController(
            train_x=train_x_mean,
            train_y=self.DATASET.train_y,
            kernel_class=PeriodicRBFKernel,
            y_std=self.DATASET.train_y_std,
        )
        shape_rx = rf"\({shape[0]}, {shape[1]}, {shape[2]}\)"
        expected_regex = (
            rf"Input data looks like it might not be rank-1, shape={shape_rx}\. If your features are "
            r"higher rank \(e\.g\. rank-2 for time series\) consider using the HigherRankFeatures "
            r"decorator on your controller and make sure that your kernel and mean functions are "
            r"defined for the rank of your input features\."
        )
        with self.assertRaisesRegex(ValueError, expected_regex):
            gp.fit(10)


class NLLTests(unittest.TestCase):
    """
    Tests for computing the NLL.
    """

    def setUp(self) -> None:
        """Code to run before each test."""

        class UniformSyntheticDataset:
            def __init__(self, function, num_train_points, num_test_points, y_std, seed=None):
                # TODO: np.random.RandomState is deprecated, use Generator API instead
                # https://github.com/gchq/Vanguard/issues/206
                self.rng = np.random.RandomState(seed)  # pylint: disable=no-member

                unscaled_train_x = self.rng.uniform(0, 1, num_train_points).reshape(-1, 1)
                scaled_train_x = (unscaled_train_x - unscaled_train_x.mean()) / unscaled_train_x.std()

                self.x = scaled_train_x
                self.y = self.rng.normal(function(self.x), y_std)

                unscaled_test_x = self.rng.uniform(0, 1, num_test_points).reshape(-1, 1)
                scaled_test_x = (unscaled_test_x - unscaled_train_x.mean()) / unscaled_train_x.std()

                self.x_test = scaled_test_x
                self.y_test = function(self.x_test)

        self.y_std = 1

        self.dataset = UniformSyntheticDataset(lambda x: np.sin(10 * x), 100, 100 // 4, self.y_std, seed=1)

        rbf_kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3))
        white_kernel = WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))
        kernel = rbf_kernel + white_kernel

        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0)
        gpr.fit(self.dataset.x, self.dataset.y)

        # generate a test set and predict on that
        z, z_std = gpr.predict(self.dataset.x_test, return_std=True)

        # get the learned hyperparameters
        params = gpr.kernel_.get_params()
        self.outputscale = params["k1__k1__constant_value"]
        self.lengthscale = params["k1__k2__length_scale"]
        self.noise_variance = params["k2__noise_level"]

        # NLL on test set
        self.sklearn_nll = self.predictive_nll(
            z.flatten(), z_std**2, self.noise_variance, self.dataset.y_test.flatten()
        )
        self.sklearn_mse = self.predictive_mse(z.flatten(), self.dataset.y_test.flatten())

    def test_gpytorch_nll(self) -> None:
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):  # pylint: disable=arguments-differ
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        train_x = torch.tensor(self.dataset.x.flatten())
        train_y = torch.tensor(self.dataset.y.flatten())
        test_x = torch.tensor(self.dataset.x_test.flatten())

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        model.likelihood.noise_covar.noise = self.noise_variance
        model.covar_module.outputscale = self.outputscale
        model.covar_module.base_kernel.lengthscale = self.lengthscale

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            _, upper = observed_pred.confidence_region()
            mean = observed_pred.mean.detach().cpu().numpy()
            std = (upper - observed_pred.mean).detach().cpu().numpy() / 2.0

            noise_variance = model.likelihood.noise.item()
            gpytorch_nll = self.predictive_nll(mean, std**2, noise_variance, self.dataset.y_test.flatten())
            gpytorch_mse = self.predictive_mse(mean, self.dataset.y_test.flatten())

        self.assertAlmostEqual(self.sklearn_nll, gpytorch_nll)
        self.assertAlmostEqual(self.sklearn_mse, gpytorch_mse)

    def test_vanguard_nll(self) -> None:
        controller = GaussianGPController(self.dataset.x, self.dataset.y, ScaledRBFKernel, y_std=self.y_std)

        controller.likelihood_noise = torch.ones_like(controller.likelihood_noise) * self.noise_variance
        controller.kernel.outputscale = self.outputscale
        controller.kernel.base_kernel.lengthscale = self.lengthscale

        posterior = controller.predictive_likelihood(self.dataset.x_test.flatten())

        vanguard_nll = posterior.nll(self.dataset.y_test.flatten(), controller.likelihood.noise.mean().item())
        vanguard_mse = posterior.mse(self.dataset.y_test.flatten())

        self.assertAlmostEqual(self.sklearn_nll, vanguard_nll, delta=1e-6)
        self.assertAlmostEqual(self.sklearn_mse, vanguard_mse, delta=1e-3)

    @staticmethod
    def predictive_nll(
        mean: np.typing.NDArray[np.floating],
        variance: np.typing.NDArray[np.floating],
        noise_variance: Union[np.typing.NDArray[np.floating], float],
        y: np.typing.NDArray[np.floating],
    ) -> np.typing.NDArray[np.floating]:
        sigma = variance + noise_variance
        rss = (y - mean) ** 2
        const = 0.5 * np.log(2 * np.pi * sigma)
        p_nll = const + rss / (2 * sigma)
        return p_nll.mean()

    @staticmethod
    def predictive_mse(
        mu_pred: np.typing.NDArray[np.floating], y: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        return ((mu_pred - y) ** 2).mean()
