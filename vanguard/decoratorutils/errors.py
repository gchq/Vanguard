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
Errors and warnings corresponding to unstable decorator combinations.

If a decorated class has implemented new functions (or overwritten existing ones)
then calling :meth:`~vanguard.decoratorutils.basedecorator.Decorator.verify_decorated_class`
will raise one of these errors or warnings.
"""


class DecoratorError(RuntimeError):
    """Base class for all decorator errors."""


class OverwrittenMethodError(DecoratorError):
    """An existing method has been overwritten."""


class UnexpectedMethodError(DecoratorError):
    """A new, unexpected method has been implemented."""


class TopmostDecoratorError(TypeError):
    """Attempting to decorate a top-level decorator."""


class MissingRequirementsError(ValueError):
    """Missing decorator requirements."""


class DecoratorWarning(RuntimeWarning):
    """Base class for all decorator warnings."""


class OverwrittenMethodWarning(DecoratorWarning):
    """An existing method has been overwritten."""


class UnexpectedMethodWarning(DecoratorWarning):
    """A new, unexpected method has been implemented."""


class BadCombinationWarning(DecoratorWarning):
    """This combination of decorators may lead to unexpected issues."""
