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
Contains any warnings that are needed in one place.
"""

import re

from gpytorch.utils.warnings import GPInputWarning, NumericalWarning
from linear_operator.utils.errors import NotPSDError

_INPUT_WARNING = "The input matches the stored training data. Did you forget to call model.train()?"
_CHOLESKY_WARNING = r"Runtime Error when computing Cholesky decomposition: (.*?)\. Using RootDecomposition\."
_JITTER_WARNING = r"A not p\.d\., added jitter of (.*?) to the diagonal"
_RE_INCORRECT_LIKELIHOOD_PARAMETER = re.compile(
    r"^.*?\.?__init__\(\) got (?:an unexpected|multiple values for) keyword argument '(.*?)'$"
)

# This is so that pre-commits don't fail on unused imports.
__all__ = [
    "GPInputWarning",
    "NumericalWarning",
    "_INPUT_WARNING",
    "_CHOLESKY_WARNING",
    "_JITTER_WARNING",
    "_RE_INCORRECT_LIKELIHOOD_PARAMETER",
    "NotPSDError",
]
