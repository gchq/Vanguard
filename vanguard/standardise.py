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
The :class:`DisableStandardScaling` decorator will disable the default input standard scaling.
"""

from typing import Any, Type, TypeVar

from vanguard.base import GPController
from vanguard.decoratorutils import Decorator, wraps_class

ControllerT = TypeVar("ControllerT", bound=GPController)


class DisableStandardScaling(Decorator):
    """
    Disable the default input scaling.

    :Example:
        >>> import numpy as np
        >>> from vanguard.kernels import ScaledRBFKernel
        >>> from vanguard.vanilla import GaussianGPController
        >>> from vanguard.standardise import DisableStandardScaling
        >>>
        >>> @DisableStandardScaling()
        ... class NoScaleController(GaussianGPController):
        ...     pass
        >>>
        >>> controller = NoScaleController(
        ...                     train_x=np.array([0, 1, 2, 3]),
        ...                     train_x_std=1,
        ...                     train_y=np.array([0, 1, 4, 9]),
        ...                     y_std=0.5,
        ...                     kernel_class=ScaledRBFKernel
        ...                     )
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialise self.

        :param kwargs: Keyword arguments passed to :class:`~vanguard.decoratorutils.basedecorator.Decorator`.
        """
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        @wraps_class(cls)
        class InnerClass(cls):
            """
            A wrapper for disabling standard scaling.
            """

            def _input_standardise_modules(self, *modules):
                return modules

        return InnerClass
