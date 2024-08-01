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
Various tools to help with the optimisation GP model parameters in Vanguard.
"""

from vanguard.optimise.finder import LearningRateFinder
from vanguard.optimise.optimiser import GreedySmartOptimiser, NoImprovementError, SmartOptimiser
from vanguard.optimise.schedule import ApplyLearningRateScheduler

__all__ = [
    "LearningRateFinder",
    "GreedySmartOptimiser",
    "NoImprovementError",
    "SmartOptimiser",
    "ApplyLearningRateScheduler",
]
