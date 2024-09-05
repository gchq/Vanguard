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
Easily enable classification within a decorator.

The return classes of all classification decorators have a distinct structure
in which the standard prediction methods are unavailable. Instead, controllers
will have :meth:`~ClassificationMixin.classify_points` and
:meth:`~ClassificationMixin.classify_fuzzy_points` which should be used.
When creating new decorators, include the :class:`ClassificationMixin` as a
mixin for the inner class, and then decorate the inner class with :class:`Classification` before returning it.
"""

import warnings
from typing import NoReturn, Tuple, Type, TypeVar, Union

import numpy as np
import numpy.typing

from vanguard.base import GPController
from vanguard.decoratorutils import Decorator, wraps_class

T = TypeVar("T")


class ClassificationMixin:
    """Mixin that provides the base methods for classification."""

    def classify_points(
        self, x: Union[float, numpy.typing.NDArray[np.floating]]
    ) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
        """
        Classify points.

        :param x: (n_predictions, n_features) The predictive inputs.
        :returns: (``predictions``, ``certainties``) where:

            * ``predictions``: (n_predictions,) The posterior predicted classes.
            * ``certainties``: (n_predictions,) The posterior predicted class probabilities.
        """
        raise NotImplementedError

    def classify_fuzzy_points(
        self, x: Union[float, numpy.typing.NDArray[np.floating]], x_std: Union[float, numpy.typing.NDArray[np.floating]]
    ) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
        """
        Classify fuzzy points.

        :param x: (n_predictions, n_features) The predictive inputs.
        :param x_std: The input noise standard deviations:

            * array_like[float]: (n_features,) The standard deviation per input dimension for the predictions,
            * float: Assume homoskedastic noise.

        :returns: (``predictions``, ``certainties``) where:

            * ``predictions``: (n_predictions,) The posterior predicted classes.
            * ``certainties``: (n_predictions,) The posterior predicted class probabilities.
        """
        raise NotImplementedError


class Classification(Decorator):
    """
    Converts a decorator class to expect a classification task.

    When used as a decorator for the output of classification decorators, this decorator automatically 'closes' the
    standard posterior methods.
    """

    def __init__(self, **kwargs):
        ignore_methods = list(kwargs.pop("ignore_methods", []))
        ignore_methods.extend(
            [
                "posterior_over_point",
                "posterior_over_fuzzy_point",
                "predictive_likelihood",
                "fuzzy_predictive_likelihood",
            ]
        )
        super().__init__(framework_class=GPController, required_decorators={}, ignore_methods=ignore_methods, **kwargs)

    def verify_decorated_class(self, cls: Type[T]) -> None:
        """
        Verify that a class can be decorated by this instance.

        :param cls: The class to be decorated.
        :raises TypeError: If cls is not a subclass of the framework_class, or if another classification decorator
            has already been applied.
        """
        super().verify_decorated_class(cls)
        if not issubclass(cls, ClassificationMixin):
            warnings.warn(
                f"Classification decorator applied to a class that doesn't "
                f"inherit from {ClassificationMixin.__name__}.",
                UserWarning,
                stacklevel=3,
                # stacklevel 2 is in BaseDecorator.__call__, so we raise this at the call site of BaseDecorator.__call__
            )
        for previous_decorator in cls.__decorators__:
            if issubclass(previous_decorator, Classification):
                msg = (
                    "This class is already decorated with a classification decorator. "
                    "Please use only one classification decorator at once."
                )
                raise TypeError(msg)

    def _decorate_class(self, cls: Type[T]) -> Type[T]:
        """Close off the prediction methods on a GP."""

        @wraps_class(cls)
        class InnerClass(cls):
            """Class that closes off the prediction methods."""

            def posterior_over_point(self, x: Union[float, numpy.typing.NDArray[np.floating]]) -> NoReturn:
                """Use :meth:`classify_points` instead."""
                raise TypeError("The 'classify_points' method should be used instead.")

            def posterior_over_fuzzy_point(
                self,
                x: Union[float, numpy.typing.NDArray[np.floating]],
                x_std: Union[float, numpy.typing.NDArray[np.floating]],
            ) -> NoReturn:
                """Use :meth:`classify_fuzzy_points` instead."""
                raise TypeError("The 'classify_fuzzy_points' method should be used instead.")

            def predictive_likelihood(self, x: Union[float, numpy.typing.NDArray[np.floating]]) -> NoReturn:
                """Use :meth:`classify_points` instead."""
                raise TypeError("The 'classify_points' method should be used instead.")

            def fuzzy_predictive_likelihood(
                self,
                x: Union[float, numpy.typing.NDArray[np.floating]],
                x_std: Union[float, numpy.typing.NDArray[np.floating]],
            ) -> NoReturn:
                """Use :meth:`classify_fuzzy_points` instead."""
                raise TypeError("The 'classify_fuzzy_points' method should be used instead.")

        return InnerClass
