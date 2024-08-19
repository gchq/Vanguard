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
Contains the DirichletMulticlassClassification decorator.
"""

from typing import Any, Tuple, Type, TypeVar, Union

import gpytorch
import numpy as np
import numpy.typing
import torch
from gpytorch.likelihoods import DirichletClassificationLikelihood
from typing_extensions import Self

from vanguard import utils
from vanguard.base import GPController
from vanguard.classification.mixin import Classification, ClassificationMixin
from vanguard.decoratorutils import Decorator, process_args, wraps_class

ControllerT = TypeVar("ControllerT", bound=GPController)
SAMPLE_DIM, TASK_DIM = 0, 2


class DirichletMulticlassClassification(Decorator):
    """
    Implements multiclass Gaussian process classification using a Dirichlet transformation.

    This decorator allows multiclass classification with exact gaussian processes.
    The implementation is based on a GPyTorch example notebook :cite:`Maddox21` and the paper :cite:`Milios18`.

    Note that in the decorated class, the `classify_points` and `classify_fuzzy_points` methods both take a
    ``num_samples`` keyword argument - we don't get a closed-form estimate for class probabilities, but instead
    approximate them with a sampling step. The ``num_samples`` parameter represents a trade-off between speed and
    accuracy, but the default value (taken from :cite:`Maddox21`) should be good enough for most purposes.

    :Example:
        >>> from gpytorch.kernels import RBFKernel, ScaleKernel
        >>> from gpytorch.likelihoods import DirichletClassificationLikelihood
        >>> import numpy as np
        >>> from vanguard.vanilla import GaussianGPController
        >>>
        >>> @DirichletMulticlassClassification(num_classes=3)
        ... class MulticlassClassifier(GaussianGPController):
        ...     pass
        >>>
        >>> class Kernel(ScaleKernel):
        ...     def __init__(self) -> None:
        ...         super().__init__(RBFKernel(batch_shape=(3,)), batch_shape=(3,))
        >>>
        >>> train_x = np.array([0, 0.1, 0.45, 0.55, 0.9, 1])
        >>> train_y = np.array([0, 0, 1, 1, 2, 2])
        >>>
        >>> gp = MulticlassClassifier(train_x, train_y, Kernel, y_std=0,
        ...                           likelihood_class=DirichletClassificationLikelihood)
        >>> loss = gp.fit(100)
        >>>
        >>> test_x = np.array([0.05, 0.5, 0.95])
        >>> predictions, probs = gp.classify_points(test_x)
        >>> predictions.tolist()
        [0, 1, 2]
    """

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        """
        Initialise self.

        :param num_classes: The number of target classes.
        """
        self.num_classes = num_classes
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        @Classification()
        @wraps_class(cls)
        class InnerClass(cls, ClassificationMixin):
            """
            A wrapper for multiclass GP classification using a Dirichlet transformation.
            """

            _y_batch_axis = 1

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)

                self.rng = utils.optional_random_generator(all_parameters_as_kwargs.pop("rng", None))

                likelihood_class = all_parameters_as_kwargs.pop("likelihood_class")
                if not issubclass(likelihood_class, DirichletClassificationLikelihood):
                    raise ValueError(
                        "The class passed to `likelihood_class` must be a subclass of "
                        f"{DirichletClassificationLikelihood.__name__} for multiclass classification."
                    )

                train_y = all_parameters_as_kwargs.pop("train_y")

                likelihood_kwargs = all_parameters_as_kwargs.pop("likelihood_kwargs", {})
                targets = torch.as_tensor(train_y, device=self.device)
                likelihood_kwargs["targets"] = targets

                try:
                    temporary_likelihood = likelihood_class(**likelihood_kwargs)
                except IndexError as exc:
                    # No is_integer check available, sadly
                    if targets.dtype.is_floating_point or targets.dtype.is_complex:
                        msg = f"For classification, train_y must be integer-valued. Got dtype={targets.dtype}."
                        raise TypeError(msg) from exc
                    raise
                transformed_targets = temporary_likelihood.transformed_targets

                @wraps_class(self.posterior_class)
                class TransposedPosterior(self.posterior_class):
                    """
                    Transpose predictions internally.

                    Dirichlet works with batch multivariate normal, so we need to reshape predictions and samples for
                    compatibility downstream.
                    """

                    def _tensor_prediction(self) -> Tuple[torch.Tensor, torch.Tensor]:
                        """Return a transposed version of the mean of the prediction."""
                        mean, covar = super()._tensor_prediction()
                        return mean.T, torch.block_diag(*covar)

                    def _tensor_sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
                        """Return a transposed version of the sample."""
                        sample = super()._tensor_sample(sample_shape=sample_shape)
                        return sample.transpose(-1, -2)

                @wraps_class(self.posterior_collection_class)
                class TransposedMonteCarloPosteriorCollection(self.posterior_collection_class):
                    """
                    Transpose predictions internally.

                    Dirichlet works with batch multivariate normal, so we need to reshape predictions and samples for
                    compatibility downstream.
                    """

                    # TODO: This method doesn't actually work!
                    #  TypeError: 'MultivariateNormal' object is not an iterator
                    #  Shouldn't expect this to work anyway, as MonteCarloPosteriorCollection doesn't support
                    #  from_mean_and_covariance in the first place
                    # https://github.com/gchq/Vanguard/issues/289
                    @classmethod
                    def from_mean_and_covariance(cls, mean: torch.Tensor, covariance: torch.Tensor) -> Self:
                        """Transpose the mean before returning."""
                        return cls(cls._make_multivariate_normal(mean.T, covariance))

                    @property
                    def condensed_distribution(self) -> gpytorch.distributions.MultivariateNormal:
                        """
                        Return the condensed distribution.

                        Return a representative distribution of the posterior, with 1-dimensional
                        mean and 2-dimensional covariance. In this case, return a distribution
                        based on the mean and covariance returned by :meth:`_tensor_prediction`.
                        """
                        mean, covar = self._tensor_prediction()
                        return self._add_jitter(self._make_multivariate_normal(mean.T, covar))

                self.posterior_class = TransposedPosterior
                self.posterior_collection_class = TransposedMonteCarloPosteriorCollection
                super().__init__(
                    train_y=transformed_targets.detach().cpu().numpy(),
                    likelihood_class=likelihood_class,
                    likelihood_kwargs=likelihood_kwargs,
                    rng=self.rng,
                    **all_parameters_as_kwargs,
                )

            def classify_points(
                self, x: Union[float, numpy.typing.NDArray[np.floating]], *, n_posterior_samples: int = 256
            ) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
                """
                Classify points.

                :param n_posterior_samples: The number of samples to take from the posterior when approximating the
                    predicted class probabilities. The default value (taken from :cite:`Maddox21`) should be good
                    enough for most purposes.

                .. note::
                    The predictions are generated from the
                    :attr:`~vanguard.base.posterior.Posterior.condensed_distribution` property of the posterior
                    in order to be consistent across collections.
                """
                posterior = super().posterior_over_point(x)
                samples = posterior._tensor_sample(torch.Size((n_posterior_samples,)))  # pylint: disable=protected-access
                pred_samples = samples.exp()
                probs = (pred_samples / pred_samples.sum(TASK_DIM, keepdim=True)).mean(SAMPLE_DIM)
                detached_probs = probs.detach().cpu().numpy()
                predictions = detached_probs.argmax(axis=1)
                return predictions, detached_probs.max(axis=1)

            def classify_fuzzy_points(
                self,
                x: Union[float, numpy.typing.NDArray[np.floating]],
                x_std: Union[float, numpy.typing.NDArray[np.floating]],
                *,
                n_posterior_samples: int = 256,
            ) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
                """
                Classify fuzzy points.

                :param n_posterior_samples: The number of samples to take from the posterior when approximating the
                    predicted class probabilities. The default value should be good enough for most purposes.

                .. note::
                    The predictions are generated from the
                    :attr:`~vanguard.base.posterior.Posterior.condensed_distribution` property of the posterior
                    in order to be consistent across collections.
                """
                posterior = super().posterior_over_fuzzy_point(x, x_std)
                samples = posterior._tensor_sample_condensed(torch.Size((n_posterior_samples,)))  # pylint: disable=protected-access
                pred_samples = samples.exp()
                probs = (pred_samples / pred_samples.sum(TASK_DIM, keepdim=True)).mean(SAMPLE_DIM)
                detached_probs = probs.detach().cpu().numpy()
                predictions = detached_probs.argmax(axis=1)
                return predictions, detached_probs.max(axis=1)

            def _loss(self, train_x: torch.Tensor, train_y: torch.Tensor) -> torch.Tensor:
                """Accounting for multiple values."""
                return super()._loss(train_x, train_y).sum()

            @staticmethod
            def _noise_transform(gamma: Union[float, numpy.typing.NDArray[np.floating]]) -> torch.Tensor:
                return torch.stack([torch.diag(torch.matmul(g, g.T)) for g in gamma], -1).squeeze().T

            @staticmethod
            def warn_normalise_y() -> None:
                """Override base warning because classification renders y normalisation irrelevant."""

        return InnerClass
