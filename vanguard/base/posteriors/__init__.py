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
Posterior classes.

Vanguard contains classes to represent posterior distributions, which are used
to encapsulate the predictive posterior of a model at some input points.
"""

from vanguard.base.posteriors.collection import MonteCarloPosteriorCollection
from vanguard.base.posteriors.posterior import Posterior

__all__ = ["MonteCarloPosteriorCollection", "Posterior"]
