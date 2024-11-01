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
Contains the _StoreInitValues metaclass.
"""

import inspect
from typing import Any, TypeVar

T = TypeVar("T")


class _StoreInitValues(type):
    """
    A metaclass to store initialisation values.

    When this metaclass is applied to a class, the parameters passed to ``__init__``
    will be stored in the :attr:`_init_params` attribute.
    """

    def __call__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        instance = super().__call__(*args, **kwargs)
        init_signature = inspect.signature(instance.__init__)
        init_params_as_kwargs = init_signature.bind_partial(*args, **kwargs).arguments
        instance._init_params = init_params_as_kwargs
        return instance
