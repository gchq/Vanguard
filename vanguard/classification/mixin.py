"""
Easily enable classification within a decorator.

The return classes of all classification decorators have a distinct structure
in which the standard prediction methods are unavailable. Instead, controllers
will have meth:`~ClassificationMixin.classify_points` and
meth:`~ClassificationMixin.classify_fuzzy_points` which should be used.
When creating new decorators, include the class:`ClassificationMixin` as a
mixin for the inner class which will be returned to enable this.
"""


class ClassificationMixin:
    """
    Converts a decorator class to expect a classification task.

    When used as a mixin for the output of classification decorators, this class
    automatically 'closes' the standard posterior methods and adds the framework
    for the meth:`classify_points` and meth:`classify_fuzzy_points` methods.
    """
    def classify_points(self, x):
        """
        Classify points.

        :param array_like[float] x: (n_preds, n_features) The predictive inputs.
        :returns: (``predictions``, ``certainties``) where:

            * ``predictions``: (n_preds,) The posterior predicted classes.
            * ``certainties``: (n_preds,) The posterior predicted class probabilities.

        :rtype: tuple[numpy.ndarray[int, float]]
        """
        raise NotImplementedError

    def classify_fuzzy_points(self, x, x_std):
        """
        Classify fuzzy points.

        :param array_like[float] x: (n_preds, n_features) The predictive inputs.
        :param array_like[float],float x_std: The input noise standard deviations:

            * array_like[float]: (n_features,) The standard deviation per input dimension for the predictions,
            * float: Assume homoskedastic noise.

        :returns: (``predictions``, ``certainties``) where:

            * ``predictions``: (n_preds,) The posterior predicted classes.
            * ``certainties``: (n_preds,) The posterior predicted class probabilities.

        :rtype: tuple[numpy.ndarray[int, float]]
        """
        raise NotImplementedError

    def posterior_over_point(self, x):
        """Use meth:`classify_points` instead."""
        raise TypeError("The 'classify_points' method should be used instead.")

    def posterior_over_fuzzy_point(self, x, x_std):
        """Use meth:`classify_fuzzy_points` instead."""
        raise TypeError("The 'classify_fuzzy_points' method should be used instead.")

    def predictive_likelihood(self, x):
        """Use meth:`classify_points` instead."""
        raise TypeError("The 'classify_points' method should be used instead.")

    def fuzzy_predictive_likelihood(self, x, x_std):
        """Use meth:`classify_fuzzy_points` instead."""
        raise TypeError("The 'classify_fuzzy_points' method should be used instead.")
