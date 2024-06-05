"""
Easily enable classification within a decorator.

The return classes of all classification decorators have a distinct structure
in which the standard prediction methods are unavailable. Instead, controllers
will have :meth:`~ClassificationMixin.classify_points` and
:meth:`~ClassificationMixin.classify_fuzzy_points` which should be used.
When creating new decorators, include the :class:`ClassificationMixin` as a
mixin for the inner class which will be returned to enable this.
"""

from typing import NoReturn, Tuple, Union

import numpy as np
import numpy.typing


# TODO: Turns this mixin into a decorator. Currently, due to method resolution order, it's not able to "close"
#  the methods it's intended to close, and so the mixin does nothing.
# https://github.com/gchq/Vanguard/issues/188
class ClassificationMixin:
    """
    Converts a decorator class to expect a classification task.

    When used as a mixin for the output of classification decorators, this class
    automatically 'closes' the standard posterior methods and adds the framework
    for the :meth:`classify_points` and :meth:`classify_fuzzy_points` methods.
    """

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

    def posterior_over_point(self, x: Union[float, numpy.typing.NDArray[np.floating]]) -> NoReturn:
        """Use :meth:`classify_points` instead."""
        raise TypeError("The 'classify_points' method should be used instead.")

    def posterior_over_fuzzy_point(
        self, x: Union[float, numpy.typing.NDArray[np.floating]], x_std: Union[float, numpy.typing.NDArray[np.floating]]
    ) -> NoReturn:
        """Use :meth:`classify_fuzzy_points` instead."""
        raise TypeError("The 'classify_fuzzy_points' method should be used instead.")

    def predictive_likelihood(self, x: Union[float, numpy.typing.NDArray[np.floating]]) -> NoReturn:
        """Use :meth:`classify_points` instead."""
        raise TypeError("The 'classify_points' method should be used instead.")

    def fuzzy_predictive_likelihood(
        self, x: Union[float, numpy.typing.NDArray[np.floating]], x_std: Union[float, numpy.typing.NDArray[np.floating]]
    ) -> NoReturn:
        """Use :meth:`classify_fuzzy_points` instead."""
        raise TypeError("The 'classify_fuzzy_points' method should be used instead.")
