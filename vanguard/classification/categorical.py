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
Contains the CategoricalClassification decorator.
"""

from typing import Any, Tuple, Type, TypeVar, Union

import numpy as np
import numpy.typing

from vanguard import utils
from vanguard.base import GPController
from vanguard.base.posteriors.posterior import Posterior
from vanguard.classification.mixin import Classification, ClassificationMixin
from vanguard.decoratorutils import Decorator, process_args, wraps_class
from vanguard.multitask import Multitask
from vanguard.variational import VariationalInference

ControllerT = TypeVar("ControllerT", bound=GPController)


class CategoricalClassification(Decorator):
    """
    Enable categorical classification with more than two classes.

    .. note::
        Although the ``y_std`` parameter is not currently used in classification, it must still be passed.
        This is likely to change in the future, and so the type must still be correct.
        Passing ``y_std=0`` is suggested.

    .. note::
        The :class:`~vanguard.variational.VariationalInference` and
        :class:`~vanguard.multitask.decorator.Multitask` decorators are required for this decorator to be applied.

    :Example:
        >>> from gpytorch.likelihoods import BernoulliLikelihood
        >>> from gpytorch.kernels import RBFKernel
        >>> from gpytorch.mlls import VariationalELBO
        >>> import numpy as np
        >>> import torch
        >>> from vanguard.vanilla import GaussianGPController
        >>> from vanguard.classification.likelihoods import MultitaskBernoulliLikelihood
        >>>
        >>> @CategoricalClassification(num_classes=3)
        ... @Multitask(num_tasks=3)
        ... @VariationalInference()
        ... class CategoricalClassifier(GaussianGPController):
        ...     pass
        >>>
        >>> train_x = np.array([0, 0.5, 0.9, 1])
        >>> train_y = np.array([[1, 0, 0], [0, 1,0], [0, 0, 1], [0, 0, 1]])
        >>> gp = CategoricalClassifier(train_x, train_y, RBFKernel, y_std=0,
        ...                            likelihood_class=MultitaskBernoulliLikelihood,
        ...                            marginal_log_likelihood_class=VariationalELBO)
        >>> loss = gp.fit(100)
        >>>
        >>> test_x = np.array([0.05, 0.95])
        >>> predictions, probs = gp.classify_points(test_x)
        >>> predictions.tolist()
        [0, 2]
    """

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        """
        Initialise self.

        :param num_classes: The number of target classes.
        :param kwargs: Keyword arguments passed to :class:`~vanguard.decoratorutils.basedecorator.Decorator`.
        """
        super().__init__(framework_class=GPController, required_decorators={VariationalInference, Multitask}, **kwargs)
        self.num_classes = num_classes

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        decorator = self

        @Classification()
        @wraps_class(cls)
        class InnerClass(cls, ClassificationMixin):
            """
            A wrapper for implementing categorical classification.
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                all_parameters_as_kwargs.pop("self")
                self.rng = utils.optional_random_generator(all_parameters_as_kwargs.pop("rng", None))

                likelihood_class = all_parameters_as_kwargs.pop("likelihood_class")
                likelihood_kwargs = all_parameters_as_kwargs.pop("likelihood_kwargs", dict())
                likelihood_kwargs["num_classes"] = decorator.num_classes
                super().__init__(
                    likelihood_class=likelihood_class,
                    likelihood_kwargs=likelihood_kwargs,
                    rng=self.rng,
                    **all_parameters_as_kwargs,
                )

            def classify_points(
                self, x: Union[float, numpy.typing.NDArray[np.floating]]
            ) -> Tuple[numpy.typing.NDArray[np.integer], Union[float, numpy.typing.NDArray[np.floating]]]:
                """Classify points."""
                predictive_likelihood = super().predictive_likelihood(x)
                return self._get_predictions_from_posterior(predictive_likelihood)

            def classify_fuzzy_points(
                self,
                x: Union[float, numpy.typing.NDArray[np.floating]],
                x_std: Union[float, numpy.typing.NDArray[np.floating]],
            ) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
                """Classify fuzzy points."""
                predictive_likelihood = super().fuzzy_predictive_likelihood(x, x_std)
                return self._get_predictions_from_posterior(predictive_likelihood)

            @staticmethod
            def _get_predictions_from_posterior(
                posterior: Posterior,
            ) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
                """
                Get predictions from a posterior distribution.

                :param posterior: The posterior distribution.
                :returns: The predicted class labels, and the certainty probabilities.
                """
                probs: numpy.typing.NDArray = posterior.distribution.probs.detach().cpu().numpy()
                if probs.ndim == 3:
                    # TODO: unsure why this is here? Document this, and then test it if it's intentional
                    # https://github.com/gchq/Vanguard/issues/234
                    probs = probs.mean(0)
                normalised_probs = probs / probs.sum(axis=-1).reshape((-1, 1))
                prediction = np.argmax(normalised_probs, axis=1)
                return prediction, np.max(normalised_probs, axis=1)

            @staticmethod
            def warn_normalise_y() -> None:
                """Override base warning because classification renders y normalisation irrelevant."""

        return InnerClass
