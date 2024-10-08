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
Contains base GPController class which is sub-classed for all GP controllers in Vanguard.
"""

import torch

import vanguard.utils
from vanguard.base.gpcontroller import GPController

torch.set_default_device(vanguard.utils.DEFAULT_DEVICE)
torch.set_default_dtype(vanguard.utils.DEFAULT_DTYPE)

__all__ = ["GPController"]
