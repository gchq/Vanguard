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
Basic end to end functionality test for warping of Gaussian processes in Vanguard.
"""

from typing import Literal, Optional, Union

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import Tensor

from tests.cases import get_default_rng
from tests.integration.util import convert_array_type, train_test_split_convert
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.warps import SetWarp, WarpFunction
from vanguard.warps.warpfunctions import (
    AffineWarpFunction,
    ArcSinhWarpFunction,
    BoxCoxWarpFunction,
    LogitWarpFunction,
    PositiveAffineWarpFunction,
    SinhWarpFunction,
    SoftPlusWarpFunction,
)

TrainTestData = Union[tuple[NDArray, NDArray, NDArray, NDArray, NDArray], tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]


class TestWarpsUsage:
    """
    A subclass of TestCase designed to check end-to-end usage of warping code.
    """

    num_train_points = 50
    num_test_points = 50
    n_sgd_iters = 10
    small_noise = 0.1

    def make_data(self, array_type: Literal["ndarray", "tensor"], warp: Optional[WarpFunction]) -> TrainTestData:
        """
        Generate data for testing.

        For SoftPlus and Logit warp functions, there are some numerical restrictions on the y-value in place, so we
        provide different data.

        :param array_type: One of "ndarray" or "tensor", depending on the desired output type.
        :param warp: An instance of WarpFunction that the data is to be generated for use with; if a SoftPlus or
            Logit warp is provided, a different set of y-values is provided to avoid numerical issues.
        :return: Tuple (x_train, y_train, y_train_std, x_test, y_test)
        """
        rng = get_default_rng()
        # Define some data.
        if isinstance(warp, SoftPlusWarpFunction):
            # For numerical reasons, we must avoid certain values of `y`. This is due to SoftPlusWarpFunction,
            # which applies the warp :math:`y\mapsto\log(e^y - 1)`, meaning we don't want `y` to grow too large or we
            # might hit numerical issues when taking the exponential, but also we need to ensure that :math:`e^y - 1`
            # does not get too close to zero or become negative. For this reason, we ensure `y` takes values around
            # 2-3 which covers both cases.
            x = np.linspace(start=4, stop=6, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
            y = np.squeeze(x / 2.0) + rng.normal(scale=self.small_noise, size=x.shape[0])
        elif isinstance(warp, LogitWarpFunction):
            # Keep `y` between 0 and 1, which ensures the logits make sense
            x = np.linspace(start=0.1, stop=1.0, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
            y = np.squeeze(x / 2.0) + rng.normal(scale=0.1 * self.small_noise, size=x.shape[0])
            y = np.clip(y, 0, 1)
        else:
            # Specifics of y-values don't matter, so take `x \sin x` for x in [0, 10].
            x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
            y = np.squeeze(x * np.sin(x)) + rng.normal(scale=self.small_noise, size=x.shape[0])

        x_train, x_test, y_train, y_test = train_test_split_convert(
            x, y, n_test_points=self.num_test_points, array_type=array_type, rng=rng
        )
        y_train_std = convert_array_type(torch.ones_like(torch.as_tensor(y_train)) * self.small_noise, array_type)

        return x_train, y_train, y_train_std, x_test, y_test

    @pytest.mark.parametrize(
        "warp_function",
        [
            pytest.param(AffineWarpFunction(), id="Affine"),
            pytest.param(PositiveAffineWarpFunction(b=6.0), id="PositiveAffine"),
            pytest.param(BoxCoxWarpFunction(lambda_=0.5), id="BoxCox"),
            pytest.param(ArcSinhWarpFunction(), id="ArcSinh"),
            pytest.param(SinhWarpFunction(), id="Sinh"),
            pytest.param(SoftPlusWarpFunction(), id="SoftPlus"),
            pytest.param(LogitWarpFunction(), id="Logit"),
        ],
    )
    @pytest.mark.parametrize("array_type", ["ndarray", "tensor"])
    def test_warps(self, warp_function: WarpFunction, array_type: Literal["ndarray", "tensor"]) -> None:
        """
        Verify Vanguard usage on a simple, single variable regression problem.

        Warping is applied, where we consider the warping functions: AffineWarpFunction,
        PositiveAffineWarpFunction, BoxCoxWarpFunction, ArcSinhWarpFunction SinhWarpFunction, and SoftPlusWarpFunction.

        Note that SoftPlusWarpFunction is tested on a different dataset to the rest, due to numerical requirements -
        see comments in `make_data`.

        We generate a single feature `x` and a continuous target `y`, and verify that a
        warped GP can be fit to this data.
        """
        x_train, y_train, y_train_std, x_test, _ = self.make_data(array_type, warp_function)

        # Define the warped controller object
        @SetWarp(warp_function, ignore_all=True)
        class WarpedController(GaussianGPController):
            pass

        # Define the controller object, with an assumed small amount of noise
        gp = WarpedController(
            train_x=x_train,
            train_y=y_train,
            kernel_class=ScaledRBFKernel,
            y_std=y_train_std,
            rng=get_default_rng(),
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        prediction_medians, prediction_ci_lower, prediction_ci_upper = gp.posterior_over_point(
            x_test
        ).confidence_interval()

        # Sense check the outputs. Note that we don't make any check on the quality of the returned intervals.
        assert not torch.any(torch.isnan(prediction_ci_lower))
        assert not torch.any(torch.isnan(prediction_medians))
        assert not torch.any(torch.isnan(prediction_ci_upper))
        assert torch.all(prediction_medians < prediction_ci_upper)
        assert torch.all(prediction_ci_lower < prediction_medians)

    @pytest.mark.parametrize(
        "warp_function",
        [pytest.param(SoftPlusWarpFunction(), id="SoftPlus"), pytest.param(LogitWarpFunction(), id="Logit")],
    )
    @pytest.mark.parametrize("array_type", ["ndarray", "tensor"])
    def test_invalid_negative_inputs(
        self, warp_function: WarpFunction, array_type: Literal["ndarray", "tensor"]
    ) -> None:
        """
        Test that passing negative y-values raises an error if the warp function doesn't allow it.

        We test logit and soft-plus warps here; the other warp functions should function fine with negative inputs.
        """
        x_train, y_train, y_train_std, _, _ = self.make_data(array_type=array_type, warp=None)

        # Define the warped controller object
        @SetWarp(warp_function, ignore_all=True)
        class WarpedController(GaussianGPController):
            pass

        # Also try to specify the gp with invalid `y` data that should not allow such warping,
        # and check an appropriate error is raised
        gp_invalid = WarpedController(
            train_x=x_train,
            train_y=-100.0 * y_train,
            kernel_class=ScaledRBFKernel,
            y_std=y_train_std,
            rng=get_default_rng(),
        )
        # TODO: check for something more specific than just `Exception`!
        # https://github.com/gchq/Vanguard/issues/401
        with pytest.raises(Exception):
            gp_invalid.fit(n_sgd_iters=self.n_sgd_iters)
