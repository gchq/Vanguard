"""
Test the behaviour of the StandardiseXModule class.
"""

import unittest

import numpy as np
import numpy.typing
import torch
from gpytorch.kernels import RBFKernel
from gpytorch.means import LinearMean

from tests.cases import get_default_rng
from vanguard.base.standardise import StandardiseXModule
from vanguard.standardise import DisableStandardScaling
from vanguard.vanilla import GaussianGPController


class StandardiseModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = torch.randn((30, 2)) * 3 + 4
        self.standardised_data = (self.data - self.data.mean(dim=0)) / self.data.std(dim=0)
        self.standardiser = StandardiseXModule.from_data(self.data, None, None)

        class BaseMean(LinearMean):
            """A test base mean."""

            def __init__(self) -> None:
                super().__init__(input_size=2)

        scaled_mean_class = self.standardiser.apply(BaseMean)
        scaled_kernel_class = self.standardiser.apply(RBFKernel)

        self.seed = torch.seed()
        self.rng = get_default_rng()
        self.base_mean = BaseMean()
        torch.manual_seed(self.seed)  # reset the seed to reproduce the random parameters
        self.scaled_mean = scaled_mean_class()

        self.base_kernel = RBFKernel()
        self.scaled_kernel = scaled_kernel_class()

    def _test_standardised_module_gives_same_results_as_plain_module_on_different_data(
        self, base_module: torch.nn.Module, scaled_module: torch.nn.Module
    ) -> None:
        standard_output = self._to_numpy(scaled_module(self.data))
        plain_output = self._to_numpy(base_module(self.standardised_data))
        np.testing.assert_array_almost_equal(standard_output, plain_output)

    def _test_standardised_module_gives_different_results_to_plain_module_on_raw_data(
        self, base_module: torch.nn.Module, scaled_module: torch.nn.Module
    ) -> None:
        standard_output = self._to_numpy(scaled_module(self.data))
        plain_output = self._to_numpy(base_module(self.data))
        np.testing.assert_array_less(-np.abs(standard_output - plain_output).mean(), -1e-6)

    def _test_standardised_module_gives_different_results_to_plain_module_on_standardised_data(
        self, base_module: torch.nn.Module, scaled_module: torch.nn.Module
    ) -> None:
        standard_output = self._to_numpy(scaled_module(self.standardised_data))
        plain_output = self._to_numpy(base_module(self.standardised_data))
        np.testing.assert_array_less(-np.abs(standard_output - plain_output).mean(), -1e-6)

    def _test_standardised_module_gives_different_results_to_plain_module_on_different_data_wrong_way(
        self, base_module: torch.nn.Module, scaled_module: torch.nn.Module
    ) -> None:
        standard_output = self._to_numpy(scaled_module(self.standardised_data))
        plain_output = self._to_numpy(base_module(self.data))
        np.testing.assert_array_less(-np.abs(standard_output - plain_output).mean(), -1e-6)

    def test_mean_and_scale(self) -> None:
        torch.testing.assert_close(self.standardiser.mean, self.data.mean(dim=0))
        torch.testing.assert_close(self.standardiser.scale, self.data.std(dim=0))

    def test_standardised_mean_gives_same_results_as_plain_mean_on_different_data(self) -> None:
        self._test_standardised_module_gives_same_results_as_plain_module_on_different_data(
            self.base_mean, self.scaled_mean
        )

    def test_standardised_mean_gives_different_results_to_plain_mean_on_raw_data(self) -> None:
        self._test_standardised_module_gives_different_results_to_plain_module_on_raw_data(
            self.base_mean, self.scaled_mean
        )

    def test_standardised_mean_gives_different_results_to_plain_mean_on_standardised_data(self) -> None:
        self._test_standardised_module_gives_different_results_to_plain_module_on_standardised_data(
            self.base_mean, self.scaled_mean
        )

    def test_standardised_mean_gives_different_results_to_plain_mean_on_different_data_wrong_way(self) -> None:
        self._test_standardised_module_gives_different_results_to_plain_module_on_different_data_wrong_way(
            self.base_mean, self.scaled_mean
        )

    def test_standardised_kernel_gives_same_results_as_plain_kernel_on_different_data(self) -> None:
        self._test_standardised_module_gives_same_results_as_plain_module_on_different_data(
            self.base_kernel, self.scaled_kernel
        )

    def test_standardised_kernel_gives_different_results_to_plain_kernel_on_raw_data(self) -> None:
        self._test_standardised_module_gives_different_results_to_plain_module_on_raw_data(
            self.base_kernel, self.scaled_kernel
        )

    def test_standardised_kernel_gives_different_results_to_plain_kernel_on_standardised_data(self) -> None:
        self._test_standardised_module_gives_different_results_to_plain_module_on_standardised_data(
            self.base_kernel, self.scaled_kernel
        )

    def test_standardised_kernel_gives_different_results_to_plain_kernel_on_different_data_wrong_way(self) -> None:
        self._test_standardised_module_gives_different_results_to_plain_module_on_different_data_wrong_way(
            self.base_kernel, self.scaled_kernel
        )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> numpy.typing.NDArray[np.floating]:
        try:
            tensor = tensor.to_dense()
        except AttributeError:
            pass
        return tensor.detach().cpu().numpy()


class DisableStandardiseModuleTests(StandardiseModuleTests):
    """
    Just replace the base mean and kernel from the above tests with those
    of a controller on which standard scaling has been disabled.
    """

    def setUp(self) -> None:
        super().setUp()

        @DisableStandardScaling()
        class DisableStandardScalingController(GaussianGPController):
            pass

        torch.manual_seed(self.seed)  # reset the seed to reproduce the random parameters
        gp = DisableStandardScalingController(
            train_x=self.data,
            train_y=self.rng.standard_normal(self.data.shape[0]),
            y_std=0,
            kernel_class=RBFKernel,
            mean_class=type(self.base_mean),
            rng=self.rng,
        )
        self.base_mean = gp.mean
        self.base_kernel = gp.kernel
