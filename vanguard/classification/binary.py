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
Contains the BinaryClassification decorator.
"""

from typing import Any, TypeVar, Union

import numpy as np
import numpy.typing
import torch
from gpytorch.likelihoods import BernoulliLikelihood
from torch import Tensor
from typing_extensions import override

from vanguard import utils
from vanguard.base import GPController
from vanguard.classification.mixin import Classification, ClassificationMixin
from vanguard.decoratorutils import Decorator, process_args, wraps_class
from vanguard.variational import VariationalInference

ControllerT = TypeVar("ControllerT", bound=GPController)


class BinaryClassification(Decorator):
    r"""
    A decorator which enables binary classification.

    .. note::
        Although the ``y_std`` parameter is not currently used in classification, it must still be passed.
        This is likely to change in the future, and so the type must still be correct.
        Passing ``y_std=0`` is suggested.

    .. note::
        When used in conjunction with the :class:`~gpytorch.likelihoods.BernoulliLikelihood` class,
        the probit likelihood is calculated in closed form by applying the following formula :cite:`Kuss05`:

        .. math::
            q(y_*=1\mid\mathcal{D},{\pmb{\theta}},{\bf x_*})
            = \int {\bf\Phi}(f_*)\mathcal{N}(f_*\mid\mu_*,\sigma_*^2)df_*
            = {\bf\Phi}\left( \frac{\mu_*}{\sqrt{1 + \sigma_*^2}} \right ).

        This means that the predictive uncertainty is taken into account.

    .. note::
        The :class:`~vanguard.variational.VariationalInference` decorator is required for this
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
        >>> gp = BinaryClassifier(train_x, train_y, ScaledRBFKernel, y_std=0.0,
        ...                       likelihood_class=BernoulliLikelihood,
        ...                       marginal_log_likelihood_class=VariationalELBO)
        >>> loss = gp.fit(100)
        >>>
        >>> test_x = np.array([0.05, 0.95])
        >>> predictions, probs = gp.classify_points(test_x)
        >>> predictions.tolist()
        [0, 1]
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialise self.

        :param kwargs: Keyword arguments passed to :class:`~vanguard.decoratorutils.basedecorator.Decorator`.
        """
        super().__init__(framework_class=GPController, required_decorators={VariationalInference}, **kwargs)

    @property
    @override
    def safe_updates(self) -> dict[type, set[str]]:
        return self._add_to_safe_updates(
            super().safe_updates,
            {VariationalInference: {"__init__", "_predictive_likelihood", "_fuzzy_predictive_likelihood"}},
        )

    def _decorate_class(self, cls: type[ControllerT]) -> type[ControllerT]:
        # We set ignore_all here as it doesn't make sense for @Classification to be checking for overrides - that
        # should only be done by decorators that actually add mathematical changes
        @Classification(ignore_all=True)
        @wraps_class(cls, decorator_source=self)
        class InnerClass(cls, ClassificationMixin):
            """
            A wrapper for implementing binary classification.
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                self.rng = utils.optional_random_generator(all_parameters_as_kwargs.pop("rng", None))

                likelihood_class = all_parameters_as_kwargs.pop("likelihood_class")
                if not issubclass(likelihood_class, BernoulliLikelihood):
                    raise ValueError(
                        "The class passed to `likelihood_class` must be a subclass "
                        f"of {BernoulliLikelihood.__name__} for binary classification."
                    )

                super().__init__(likelihood_class=likelihood_class, rng=self.rng, **all_parameters_as_kwargs)

            def classify_points(
                self, x: Union[float, numpy.typing.NDArray[np.floating], Tensor]
            ) -> tuple[Tensor, Tensor]:
                """Classify points."""
                x = torch.as_tensor(x)
                means_as_floats, _ = super().predictive_likelihood(x).prediction()
                return self._get_predictions_from_prediction_means(means_as_floats)

            def classify_fuzzy_points(
                self,
                x: Union[float, numpy.typing.NDArray[np.floating], Tensor],
                x_std: Union[float, numpy.typing.NDArray[np.floating], Tensor],
            ) -> tuple[Tensor, Tensor]:
                """Classify fuzzy points."""
                x = torch.as_tensor(x)
                x_std = torch.as_tensor(x_std)
                means_as_floats, _ = super().fuzzy_predictive_likelihood(x, x_std).prediction()
                return self._get_predictions_from_prediction_means(means_as_floats)

            @staticmethod
            def _get_predictions_from_prediction_means(
                means: Union[float, numpy.typing.NDArray[np.floating], Tensor],
            ) -> tuple[Tensor, Tensor]:
                """
                Get the predictions and certainty probabilities from predictive likelihood means.

                :param means: The prediction means in the range [0, 1].
                :returns: The predicted class labels, and the certainty probabilities.
                """
                prediction = torch.as_tensor(means).round().to(torch.int)
                certainty = torch.maximum(means, 1 - means)
                return prediction, certainty

            @staticmethod
            def warn_normalise_y() -> None:
                """Override base warning because classification renders y normalisation irrelevant."""

        return InnerClass
