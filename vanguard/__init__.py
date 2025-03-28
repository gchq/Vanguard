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
A Python package for advanced GP usage.

Vanguard contains convenience wrappers for a number of advanced Gaussian Process
techniques, designed with a focus on simplicity, extension and combination.
"""

import torch

import vanguard.utils
from vanguard import _bibliography

torch.set_default_dtype(vanguard.utils.DEFAULT_DTYPE)
torch.set_default_device(vanguard.utils.DEFAULT_DEVICE)


__author__ = "GCHQ"
__version__ = "3.1.0"
__bibliography__ = _bibliography.find_bibliography()
