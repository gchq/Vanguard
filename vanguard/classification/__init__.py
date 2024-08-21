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
It is possible to convert a regression problem into a classification problem, allowing the use of Gaussian processes.
"""

from vanguard.classification.binary import BinaryClassification
from vanguard.classification.categorical import CategoricalClassification
from vanguard.classification.dirichlet import DirichletMulticlassClassification

__all__ = [
    "BinaryClassification",
    "CategoricalClassification",
    "DirichletMulticlassClassification",
]
