"""
Contains the BinaryClassification decorator.
"""
from typing import Tuple, Type, TypeVar, Union

import numpy as np
import numpy.typing
from gpytorch.likelihoods import BernoulliLikelihood

from ..base import GPController
from ..decoratorutils import Decorator, process_args, wraps_class
from ..variational import VariationalInference
from .mixin import ClassificationMixin

ControllerT = TypeVar("ControllerT", bound=GPController)


class BinaryClassification(Decorator):
    r"""
    A decorator which enables binary classification.

    .. note::
        Although the ``y_std`` parameter is not currently used in classification, it must still be passed.
        This is likely to change in the future, and so the type must still be correct.
        Passing ``y_std=0`` is suggested.

    .. note::
        When used in conjunction with the class:`~gpytorch.likelihoods.BernoulliLikelihood` class,
        the probit likelihood is calculated in closed form by applying the following formula :cite:`Kuss05`:

        .. math::
            q(y_*=1\mid\mathcal{D},{\pmb{\theta}},{\bf x_*})
            = \int {\bf\Phi}(f_*)\mathcal{N}(f_*\mid\mu_*,\sigma_*^2)df_*
            = {\bf\Phi}\left( \frac{\mu_*}{\sqrt{1 + \sigma_*^2}} \right ).

        This means that the predictive uncertainty is taken into account.

    .. note::
        The class:`~vanguard.variational.VariationalInference` decorator is required for this
        decorator to be applied.

    :Example:
        >>> from gpytorch.likelihoods import BernoulliLikelihood
        >>> from gpytorch.mlls import VariationalELBO
        >>> import numpy as np
        >>> from vanguard.kernels import ScaledRBFKernel
        >>> from vanguard.vanilla import GaussianGPController
        >>>
        >>> @BinaryClassification()
        ... @VariationalInference()
        ... class BinaryClassifier(GaussianGPController):
        ...     pass
        >>>
        >>> train_x = np.array([0, 0.1, 0.9, 1])
        >>> train_y = np.array([0, 0, 1, 1])
        >>>
        >>> gp = BinaryClassifier(train_x, train_y, ScaledRBFKernel, y_std=0,
        ...                       likelihood_class=BernoulliLikelihood,
        ...                       marginal_log_likelihood_class=VariationalELBO)
        >>> loss = gp.fit(100)
        >>>
        >>> test_x = np.array([0.05, 0.95])
        >>> preds, probs = gp.classify_points(test_x)
        >>> preds
        array([0, 1])
    """
    def __init__(self, **kwargs):
        """
        Initialise self.

        :param kwargs: Keyword arguments passed to class:`~vanguard.decoratorutils.basedecorator.Decorator`.
        """
        super().__init__(framework_class=GPController, required_decorators={VariationalInference}, **kwargs)

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        @wraps_class(cls)
        class InnerClass(cls, ClassificationMixin):
            """
            A wrapper for implementing binary classification.
            """
            def __init__(self, *args, **kwargs):

                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                all_parameters_as_kwargs.pop("self")

                likelihood_class = all_parameters_as_kwargs.pop("likelihood_class")
                if not issubclass(likelihood_class, BernoulliLikelihood):
                    raise ValueError("The class passed to `likelihood_class` must be a subclass "
                                     f"of {BernoulliLikelihood.__name__} for binary classification.")

                super().__init__(likelihood_class=likelihood_class, **all_parameters_as_kwargs)

            def classify_points(self, x: Union[float, numpy.typing.NDArray[np.floating]]) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
                """Classify points."""
                means_as_floats, _ = super().predictive_likelihood(x).prediction()
                return self._get_predictions_from_prediction_means(means_as_floats)

            def classify_fuzzy_points(
                    self, x: Union[float, numpy.typing.NDArray[np.floating]], x_std: Union[float, numpy.typing.NDArray[np.floating]]
            ) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
                """Classify fuzzy points."""
                means_as_floats, _ = super().fuzzy_predictive_likelihood(x, x_std).prediction()
                return self._get_predictions_from_prediction_means(means_as_floats)

            @staticmethod
            def _get_predictions_from_prediction_means(
                    means: Union[float, numpy.typing.NDArray[np.floating]]
            ) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
                """
                Get the predictions and certainty probabilities from predictive likelihood means.

                :param means: The prediction means in the range [0, 1].
                :returns: The predicted class labels, and the certainty probabilities.
                """
                prediction = means.round().astype(int)
                certainty = np.maximum(means, 1 - means)
                return prediction, certainty

            @staticmethod
            def warn_normalise_y() -> None:
                """Override base warning because classification renders y normalisation irrelevant."""
                pass

        return InnerClass
