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
    """
    Tests relating to standardisation of input data using the `StandardiseXModule` class.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
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
        # Reset the seed to reproduce the random parameters
        torch.manual_seed(self.seed)
        self.scaled_mean = scaled_mean_class()

        self.base_kernel = RBFKernel()
        self.scaled_kernel = scaled_kernel_class()

    def _test_standardised_module_gives_same_results_as_plain_module_on_different_data(
        self, base_module: torch.nn.Module, scaled_module: torch.nn.Module
    ) -> None:
        """
        Test usage of scaled and unscaled modules on corresponding data.

        Test that calling the scaled module on raw data agrees with calling the unscaled module on
        scaled data. Here we expect the scaled module to internally scale the data before computations,
        whereas the unscaled module should not apply any scaling, but since the data is already scaled,
        the results should be the same.
        """
        standard_output = self._to_numpy(scaled_module(self.data))
        plain_output = self._to_numpy(base_module(self.standardised_data))
        np.testing.assert_array_almost_equal(standard_output, plain_output)

    def _test_standardised_module_gives_different_results_to_plain_module_on_raw_data(
        self, base_module: torch.nn.Module, scaled_module: torch.nn.Module
    ) -> None:
        """
        Test that a scaled and unscaled module give different results on the same input data.

        Here we pass a scaled module and an unscaled module the same (unscaled) data and expect
        different results, since one should apply scaling before computations and one should not.
        """
        standard_output = self._to_numpy(scaled_module(self.data))
        plain_output = self._to_numpy(base_module(self.data))
        np.testing.assert_array_less(-np.abs(standard_output - plain_output).mean(), -1e-6)

    def _test_standardised_module_gives_different_results_to_plain_module_on_standardised_data(
        self, base_module: torch.nn.Module, scaled_module: torch.nn.Module
    ) -> None:
        """
        Test that a scaled and unscaled module give different results on the same input data.

        Here we pass a scaled module and an unscaled module the same (scaled) data and expect
        different results, since one should apply scaling before computations and one should not.
        """
        standard_output = self._to_numpy(scaled_module(self.standardised_data))
        plain_output = self._to_numpy(base_module(self.standardised_data))
        np.testing.assert_array_less(-np.abs(standard_output - plain_output).mean(), -1e-6)

    def _test_standardised_module_gives_different_results_to_plain_module_on_different_data_wrong_way(
        self, base_module: torch.nn.Module, scaled_module: torch.nn.Module
    ) -> None:
        """
        Test that a scaled and unscaled module give different results on different input data.

        Here we pass a scaled module an already scaled dataset, and an unscaled module an unscaled dataset.
        In this case, we would not expect the outputs to match.
        """
        standard_output = self._to_numpy(scaled_module(self.standardised_data))
        plain_output = self._to_numpy(base_module(self.data))
        np.testing.assert_array_less(-np.abs(standard_output - plain_output).mean(), -1e-6)

    def test_mean_and_scale(self) -> None:
        """Test that the mean and standard deviation of the standardiser match those of the data."""
        torch.testing.assert_close(self.standardiser.mean, self.data.mean(dim=0))
        torch.testing.assert_close(self.standardiser.scale, self.data.std(dim=0))

    def test_standardised_mean_gives_same_results_as_plain_mean_on_different_data(self) -> None:
        """Test mean modules give expected outputs on different data."""
        self._test_standardised_module_gives_same_results_as_plain_module_on_different_data(
            self.base_mean, self.scaled_mean
        )

    def test_standardised_mean_gives_different_results_to_plain_mean_on_raw_data(self) -> None:
        """Test mean modules give expected outputs on unscaled data."""
        self._test_standardised_module_gives_different_results_to_plain_module_on_raw_data(
            self.base_mean, self.scaled_mean
        )

    def test_standardised_mean_gives_different_results_to_plain_mean_on_standardised_data(self) -> None:
        """Test mean modules give expected outputs on scaled data."""
        self._test_standardised_module_gives_different_results_to_plain_module_on_standardised_data(
            self.base_mean, self.scaled_mean
        )

    def test_standardised_mean_gives_different_results_to_plain_mean_on_different_data_wrong_way(self) -> None:
        """Test mean modules give expected outputs on different data."""
        self._test_standardised_module_gives_different_results_to_plain_module_on_different_data_wrong_way(
            self.base_mean, self.scaled_mean
        )

    def test_standardised_kernel_gives_same_results_as_plain_kernel_on_different_data(self) -> None:
        """Test mean modules give expected outputs on different data."""
        self._test_standardised_module_gives_same_results_as_plain_module_on_different_data(
            self.base_kernel, self.scaled_kernel
        )

    def test_standardised_kernel_gives_different_results_to_plain_kernel_on_raw_data(self) -> None:
        """Test mean modules give expected outputs on unscaled data."""
        self._test_standardised_module_gives_different_results_to_plain_module_on_raw_data(
            self.base_kernel, self.scaled_kernel
        )

    def test_standardised_kernel_gives_different_results_to_plain_kernel_on_standardised_data(self) -> None:
        """Test mean modules give expected outputs on scaled data."""
        self._test_standardised_module_gives_different_results_to_plain_module_on_standardised_data(
            self.base_kernel, self.scaled_kernel
        )

    def test_standardised_kernel_gives_different_results_to_plain_kernel_on_different_data_wrong_way(self) -> None:
        """Test mean modules give expected outputs on different data."""
        self._test_standardised_module_gives_different_results_to_plain_module_on_different_data_wrong_way(
            self.base_kernel, self.scaled_kernel
        )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> numpy.typing.NDArray[np.floating]:
        """
        Convert a torch tensor to a numpy array.

        :param tensor: Torch tensor to convert
        :return: The provided tensor converted to a numpy array
        """
        try:
            tensor = tensor.to_dense()
        except AttributeError:
            pass
        return tensor.detach().cpu().numpy()


class DisableStandardiseModuleTests(StandardiseModuleTests):
    """
    Tests for functionality to disable scaling of data.

    Within this, we replace the base mean and kernel from `StandardiseModuleTests` with those
    of a controller on which standard scaling has been disabled.
    """

    def setUp(self) -> None:
        """
        Setup test by defining a controller object that disables data scaling.
        """
        super().setUp()

        @DisableStandardScaling()
        class DisableStandardScalingController(GaussianGPController):
            pass

        torch.manual_seed(self.seed)  # Reset the seed to reproduce the random parameters
        gp = DisableStandardScalingController(
            train_x=self.data,
            train_y=self.rng.standard_normal(self.data.shape[0]),
            y_std=0.0,
            kernel_class=RBFKernel,
            mean_class=type(self.base_mean),
            rng=self.rng,
        )
        self.base_mean = gp.mean
        self.base_kernel = gp.kernel
