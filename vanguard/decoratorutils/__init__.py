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
Tools to support decorators in Vanguard.

In Vanguard, decorators allow for easy, dynamic subclassing of
class:`~vanguard.base.gpcontroller.GPController` instances, to add new functionality
in an easily composable way. All new decorators should subclass from
class:`~basedecorator.Decorator` or :class:`~basedecorator.TopMostDecorator`.
See :doc:`../examples/decorator_walkthrough` for more details.
"""

from vanguard.decoratorutils.basedecorator import Decorator, TopMostDecorator
from vanguard.decoratorutils.wrapping import process_args, wraps_class

__all__ = [
    "Decorator",
    "TopMostDecorator",
    "process_args",
    "wraps_class",
]
