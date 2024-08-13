"""
Tests for the GPController class.
"""

import unittest
from typing import Callable, Union

import gpytorch
import numpy as np
import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from tests.cases import VanguardTestCase, get_default_rng
from vanguard.base import GPController
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import PeriodicRBFKernel, ScaledRBFKernel
from vanguard.optimise import SmartOptimiser
from vanguard.vanilla import GaussianGPController


class DefaultTensorTypeTests(unittest.TestCase):
    """
    Tests for the setting of the default tensor types.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.original_dtype = GaussianGPController.get_default_tensor_dtype()
        self.original_device = GaussianGPController.get_default_tensor_device()

        if self.original_dtype == torch.double:
            self.skipTest("Skipping this test because the default tensor dtype would not be changed.")

        original_tensor = torch.tensor([])
        self.assertEqual(original_tensor.dtype, self.original_dtype)
        self.assertEqual(original_tensor.device, self.original_device)

        class NewController(GaussianGPController):
            pass

        self.new_controller_class = NewController
        self.new_controller_class.set_default_tensor_dtype(torch.double)

    def tearDown(self) -> None:
        """Code to run after each test."""
        # Set the NewController's default tensor type as the original GaussianGPController's
        self.new_controller_class.set_default_tensor_dtype(self.original_dtype)
        tensor = torch.tensor([])
        self.assertEqual(tensor.dtype, self.original_dtype)
        self.assertEqual(tensor.device, self.original_device)

    def test_class_default_tensor(self) -> None:
        """Test that the new controller's default tensor dtype was set correctly to torch.DoubleTensor in setUp()."""
        self.assertEqual(self.new_controller_class.get_default_tensor_dtype(), torch.double)

    def test_superclass_default_tensor(self) -> None:
        """Test that the GaussianGPController's default tensor dtype is unchanged by setUp()."""
        self.assertEqual(GaussianGPController.get_default_tensor_dtype(), self.original_dtype)

    def test_default_tensor(self) -> None:
        """
        Test that the properties of a newly-created tensor are as expected.

        This test fails unless the tensor's dtype is double. By default, PyTorch creates tensors with float32 dtype.
        This test checks that the default tensor dtype is successfully set to torch.double in setUp() above. Note,
        in BaseGPController, we set _default_tensor_dtype = torch.float.

        This test expects the new tensor to be on the CPU if the CUDA device (i.e., GPU) is not available. The is_cuda
        property returns True if the tensor is stored on the GPU, and False otherwise.
        """
        new_tensor = torch.tensor([])
        self.assertEqual(new_tensor.dtype, torch.double)
        self.assertEqual(new_tensor.is_cuda, torch.cuda.is_available())


class InputTests(VanguardTestCase):
    """
    Tests for checking the behaviour of GP controllers.

    GP controllers are forgiving about the shape of data arrays, where possible. These tests check this behaviour.
    """

    DATASET = SyntheticDataset(rng=get_default_rng())

    def test_unsqueeze_y(self) -> None:
        """
        Make sure the tensor wrangling works if we pass y with the wrong shape.

        This test reshapes the training outputs to exactly one column (and as many rows as needed). This is then passed
        to a Vanguard GPController. We assert that storing this attribute on the controller does not affect the original
        data.
        """
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
            rng=get_default_rng(),
        )
        # Convert train_y on GPController to a NumPy array, ensuring it's on CPU and detached from the computation graph
        gp_train_y = gp.train_y.detach().cpu().numpy()
        np.testing.assert_array_almost_equal(squeezed_train_y, gp_train_y, decimal=5)

    def test_unsqueeze_x(self) -> None:
        """
        Check the tensor wrangling works if we pass x with the wrong shape.

        This test flattens the training input (i.e., reshaping to be exactly one row). This is then passed to a Vanguard
        GPController. We assert that storing this attribute on the controller does not affect the original data.
        """
        flattened_train_x = self.DATASET.train_x.ravel()
        gp = GPController(
            train_x=flattened_train_x,
            train_y=self.DATASET.train_y,
            kernel_class=PeriodicRBFKernel,
            mean_class=ConstantMean,
            y_std=self.DATASET.train_y_std,
            likelihood_class=FixedNoiseGaussianLikelihood,
            marginal_log_likelihood_class=ExactMarginalLogLikelihood,
            optimiser_class=torch.optim.Adam,
            smart_optimiser_class=SmartOptimiser,
            rng=get_default_rng(),
        )
        # Convert train_x on GPController to a NumPy array, ensuring it's on CPU and detached from the computation graph
        gp_train_x = gp.train_x.detach().cpu().numpy()
        np.testing.assert_array_almost_equal(self.DATASET.train_x, gp_train_x, decimal=5)

    def test_error_handling_of_higher_rank_features(self) -> None:
        """Test that shape errors, due to incorrectly treated high-rank features, are caught and explained."""
        shape = (len(self.DATASET.train_y), 31, 4)
        rng = get_default_rng()
        random_train_x = rng.standard_normal(shape)
        gp = GaussianGPController(
            train_x=random_train_x,
            train_y=self.DATASET.train_y,
            kernel_class=PeriodicRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=rng,
        )
        shape_rx = rf"\({shape[0]}, {shape[1]}, {shape[2]}\)"
        expected_regex = (
            rf"Input data looks like it might not be rank-1, shape={shape_rx}\. If your features are "
            r"higher rank \(e\.g\. rank-2 for time series\) consider using the HigherRankFeatures "
            r"decorator on your controller and make sure that your kernel and mean functions are "
            r"defined for the rank of your input features\."
        )
        with self.assertRaisesRegex(ValueError, expected_regex):
            gp.fit()

    @unittest.skip  # TODO: fix test; underlying issues in batch mode
    # https://github.com/gchq/Vanguard/issues/265
    def test_error_handling_of_batch_size(self) -> None:
        """Test that a UserWarning is raised when both batch_size and gradient_every are not None."""
        gp = GaussianGPController(
            train_x=self.DATASET.train_x,
            train_y=self.DATASET.train_y,
            kernel_class=PeriodicRBFKernel,
            y_std=self.DATASET.train_y_std,
            batch_size=20,
        )
        gradient_every = 2
        gp.fit()
        with self.assertWarns(UserWarning):
            gp.fit(n_sgd_iters=10, gradient_every=gradient_every)


class NLLTests(unittest.TestCase):
    """
    Tests for computing the negative log-likelihood (NLL).
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.rng = get_default_rng()

        class UniformSyntheticDataset:
            def __init__(
                self,
                function: Callable,
                num_train_points: int,
                num_test_points: int,
                y_std: Union[float, NDArray[np.floating]],
                rng: np.random.Generator,
            ) -> None:
                self.rng = rng

                unscaled_train_x = self.rng.uniform(0, 1, num_train_points).reshape(-1, 1)
                scaled_train_x = (unscaled_train_x - unscaled_train_x.mean()) / unscaled_train_x.std()

                self.x = scaled_train_x
                self.y = self.rng.normal(function(self.x), y_std)

                unscaled_test_x = self.rng.uniform(0, 1, num_test_points).reshape(-1, 1)
                scaled_test_x = (unscaled_test_x - unscaled_train_x.mean()) / unscaled_train_x.std()

                self.x_test = scaled_test_x
                self.y_test = function(self.x_test)

        self.y_std = 1

        self.dataset = UniformSyntheticDataset(lambda x: np.sin(10 * x), 100, 100 // 4, self.y_std, rng=self.rng)

        rbf_kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3))
        white_kernel = WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))
        kernel = rbf_kernel + white_kernel

        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0)
        gpr.fit(self.dataset.x, self.dataset.y)

        # Generate a test set and predict on that (with sklearn)
        z, z_std = gpr.predict(self.dataset.x_test, return_std=True)

        # Get the learned hyperparameters
        params = gpr.kernel_.get_params()
        self.outputscale = params["k1__k1__constant_value"]
        self.lengthscale = params["k1__k2__length_scale"]
        self.noise_variance = params["k2__noise_level"]

        # Get the NLL for this test set
        self.sklearn_nll = self.predictive_nll(
            mean=z.flatten(), variance=z_std**2, noise_variance=self.noise_variance, y=self.dataset.y_test.flatten()
        )
        # Get the MSE for this test set
        self.sklearn_mse = self.predictive_mse(mu_pred=z.flatten(), y=self.dataset.y_test.flatten())

    def test_gpytorch_nll(self) -> None:
        """Test that the NLL calculated with GPyTorch agrees with sklearn."""

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(
                self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.likelihood
            ) -> None:
                super().__init__(train_inputs=train_x, train_targets=train_y, likelihood=likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):  # pylint: disable=arguments-differ
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # Flatten and tensor-ify the test data
        train_x = torch.tensor(self.dataset.x.flatten())
        train_y = torch.tensor(self.dataset.y.flatten())
        test_x = torch.tensor(self.dataset.x_test.flatten())

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        model.likelihood.noise_covar.noise = self.noise_variance
        model.covar_module.outputscale = self.outputscale
        model.covar_module.base_kernel.lengthscale = self.lengthscale

        # Evaluate
        model.eval()
        likelihood.eval()

        # Disable gradient calculation and enable fast predictive variances, to enhance performance
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            _, upper = observed_pred.confidence_region()
            mean = observed_pred.mean.detach().cpu().numpy()
            std = (upper - observed_pred.mean).detach().cpu().numpy() / 2.0

            noise_variance = model.likelihood.noise.item()
            gpytorch_nll = self.predictive_nll(
                mean=mean, variance=std**2, noise_variance=noise_variance, y=self.dataset.y_test.flatten()
            )
            gpytorch_mse = self.predictive_mse(mu_pred=mean, y=self.dataset.y_test.flatten())

        self.assertAlmostEqual(self.sklearn_nll, gpytorch_nll)
        self.assertAlmostEqual(self.sklearn_mse, gpytorch_mse)

    def test_vanguard_nll(self) -> None:
        """Test that the NLL calculated with Vanguard agrees with sklearn."""
        controller = GaussianGPController(
            train_x=self.dataset.x, train_y=self.dataset.y, kernel_class=ScaledRBFKernel, y_std=self.y_std, rng=self.rng
        )

        controller.likelihood_noise = torch.ones_like(controller.likelihood_noise) * self.noise_variance
        controller.kernel.outputscale = self.outputscale
        controller.kernel.base_kernel.lengthscale = self.lengthscale

        posterior = controller.predictive_likelihood(x=self.dataset.x_test.flatten())

        vanguard_nll = posterior.nll(
            y=self.dataset.y_test.flatten(), noise_variance=controller.likelihood.noise.mean().item()
        )
        vanguard_mse = posterior.mse(y=self.dataset.y_test.flatten())

        self.assertAlmostEqual(self.sklearn_nll, vanguard_nll, delta=5e-4)
        self.assertAlmostEqual(self.sklearn_mse, vanguard_mse, delta=1e-3)

    @staticmethod
    def predictive_nll(
        mean: np.typing.NDArray[np.floating],
        variance: np.typing.NDArray[np.floating],
        noise_variance: Union[np.typing.NDArray[np.floating], float],
        y: np.typing.NDArray[np.floating],
    ) -> np.typing.NDArray[np.floating]:
        """
        Get the mean negative log-likelihood, for testing purposes.

        :param mean: The mean values of the predictive distribution.
        :param variance: The variance of the predictive distribution.
        :param noise_variance: The noise variance, as an array or a single float.
        :param y: The observed values.
        :returns: The mean negative log-likelihood of the predictive distribution.
        """
        sigma = variance + noise_variance
        rss = (y - mean) ** 2
        const = 0.5 * np.log(2 * np.pi * sigma)
        p_nll = const + rss / (2 * sigma)
        return p_nll.mean()

    @staticmethod
    def predictive_mse(
        mu_pred: np.typing.NDArray[np.floating], y: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """
        Get the mean squared error, for testing purposes.

        :param mu_pred: The mean values of the predictive distribution.
        :param y: The observed values.
        :returns: The mean squared error of the predictive distribution.
        """
        return ((mu_pred - y) ** 2).mean()
