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
Tests for kernels.
"""

import unittest

import torch

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import TimeSeriesKernel
from vanguard.vanilla import GaussianGPController


class BasicTests(unittest.TestCase):
    """
    Basic tests for the TimeSeriesKernel decorator.
    """

    def test_trains_time_feature_only(self) -> None:  # noqa: D102
        # TODO: Add a docstring explaining what this test is testing, then remove `noqa: D102`.
        # https://github.com/gchq/Vanguard/issues/445

        rng = get_default_rng()
        dataset = SyntheticDataset(rng=rng)
        controller = GaussianGPController(
            dataset.train_x, dataset.train_y, TimeSeriesKernel, y_std=dataset.train_y_std, rng=rng
        )
        controller.fit(10)
        mean, _, upper = controller.posterior_over_point(dataset.test_x).confidence_interval()
        assert torch.all(upper - mean < 0.5)
