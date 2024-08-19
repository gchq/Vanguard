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
Enable training on non-Gaussian observation noise with warping.

Warp functions are used to map data to a different domain to train the Gaussian
process. Although Vanguard contains many pre-written warp functions, any new
ones can be created by subclassing :class:`~basefunction.WarpFunction` and
implementing the :meth:`~basefunction.WarpFunction.forward`,
:meth:`~basefunction.WarpFunction.inverse` and (optionally)
:meth:`~basefunction.WarpFunction.deriv` methods.

Warp functions are applied to a :class:`~vanguard.base.gpcontroller.GPController`
subclass using the :class:`SetWarp` decorator.
"""

from vanguard.warps.basefunction import MultitaskWarpFunction, WarpFunction
from vanguard.warps.decorator import SetWarp
from vanguard.warps.distribution import WarpedGaussian
from vanguard.warps.input import SetInputWarp
from vanguard.warps.intermediate import require_controller_input

__all__ = [
    "MultitaskWarpFunction",
    "WarpFunction",
    "SetWarp",
    "WarpedGaussian",
    "SetInputWarp",
    "require_controller_input",
]
