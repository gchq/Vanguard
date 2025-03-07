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

from typing import Any, TypeVar

from typing_extensions import override

from vanguard.base import GPController
from vanguard.classification.mixin import Classification, ClassificationMixin
from vanguard.decoratorutils import Decorator, wraps_class
from vanguard.variational import VariationalInference

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
        ...                     train_x=np.array([0.0, 1.0, 2.0, 3.0]),
        ...                     train_x_std=1.0,
        ...                     train_y=np.array([0.0, 1.0, 4.0, 9.0]),
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

    @property
    @override
    def safe_updates(self) -> dict[type, set[str]]:
        return self._add_to_safe_updates(
            super().safe_updates,
            {
                ClassificationMixin: {"classify_points", "classify_fuzzy_points"},
                Classification: {
                    "posterior_over_point",
                    "posterior_over_fuzzy_point",
                    "fuzzy_predictive_likelihood",
                    "predictive_likelihood",
                },
                VariationalInference: {"__init__", "_predictive_likelihood", "_fuzzy_predictive_likelihood"},
            },
        )

    def _decorate_class(self, cls: type[ControllerT]) -> type[ControllerT]:
        @wraps_class(cls, decorator_source=self)
        class InnerClass(cls):
            """
            A wrapper for disabling standard scaling.
            """

            def _input_standardise_modules(self, *modules):
                return modules

        return InnerClass
