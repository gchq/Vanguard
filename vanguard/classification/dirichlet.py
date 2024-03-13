"""
Contains the DirichletMulticlassClassification decorator.
"""
from gpytorch.likelihoods import DirichletClassificationLikelihood
import torch
import gpytorch
import numpy as np
import numpy.typing

from ..base import GPController
from ..decoratorutils import Decorator, process_args, wraps_class
from .mixin import ClassificationMixin

from typing_extensions import Self
from typing import TypeVar, Type, NoReturn


ControllerT = TypeVar("ControllerT", bound=GPController)
SAMPLE_DIM, TASK_DIM = 0, 2


class DirichletMulticlassClassification(Decorator):
    """
    Implements multiclass Gaussian process classification using a Dirichlet transformation.

    This decorator allows multiclass classification with exact gaussian processes.
    The implementation is based on a GPyTorch example notebook :cite:`Maddox21` and the paper :cite:`Milios18`.

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
        ...     def __init__(self):
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
        >>> preds, probs = gp.classify_points(test_x)
        >>> preds
        array([0, 1, 2])
    """
    def __init__(self, num_classes: int, **kwargs):
        """
        Initialise self.

        :param num_classes: The number of target classes.
        """
        self.num_classes = num_classes
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: Type[ControllerT]) -> ControllerT:
        @wraps_class(cls)
        class InnerClass(cls, ClassificationMixin):
            """
            A wrapper for implementing variational inference.
            """
            _y_batch_axis = 1

            def __init__(self, *args, **kwargs):

                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                all_parameters_as_kwargs.pop("self")

                likelihood_class = all_parameters_as_kwargs.pop("likelihood_class")
                if not issubclass(likelihood_class, DirichletClassificationLikelihood):
                    raise ValueError("The class passed to `likelihood_class` must be a subclass of "
                                     f"{DirichletClassificationLikelihood.__name__} for binary classification.")

                train_y = all_parameters_as_kwargs.pop("train_y")

                likelihood_kwargs = all_parameters_as_kwargs.pop("likelihood_kwargs", {})
                targets = torch.as_tensor(train_y, device=self.device, dtype=torch.int64)
                likelihood_kwargs["targets"] = targets

                temporary_likelihood = likelihood_class(**likelihood_kwargs)
                transformed_targets = temporary_likelihood.transformed_targets

                @wraps_class(self.posterior_class)
                class TransposedPosterior(self.posterior_class):
                    """
                    Transpose predictions internally.

                    Dirichlet works with batch multivariate normal, so we need to reshape predictions and samples for
                    compatibility downstream.
                    """
                    def _tensor_prediction(self) -> tuple[torch.Tensor, torch.Tensor]:
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

                super().__init__(train_y=transformed_targets.detach().cpu().numpy(), likelihood_class=likelihood_class,
                                 likelihood_kwargs=likelihood_kwargs, **all_parameters_as_kwargs)

            def classify_points(self, x: numpy.typing.ArrayLike[float]) -> tuple[np.ndarray[int], np.ndarray[float]]:
                """
                Classify points.

                .. note::
                    The predictions are generated from the
                    :attr:`~vanguard.base.posterior.Posterior.condensed_distribution` property of the posterior
                    in order to be consistent across collections.
                """
                posterior = super().posterior_over_point(x)
                samples = posterior._tensor_sample(torch.Size((256,)))
                pred_samples = samples.exp()
                probs = (pred_samples / pred_samples.sum(TASK_DIM, keepdim=True)).mean(SAMPLE_DIM)
                detached_probs = probs.detach().cpu().numpy()
                predictions = detached_probs.argmax(axis=1)
                return predictions, detached_probs.max(axis=1)

            def classify_fuzzy_points(self, x: numpy.typing.ArrayLike[float], x_std: numpy.typing.ArrayLike[float]) -> tuple[np.ndarray[int], np.ndarray[float]]:
                """
                Classify fuzzy points.

                .. note::
                    The predictions are generated from the
                    :attr:`~vanguard.base.posterior.Posterior.condensed_distribution` property of the posterior
                    in order to be consistent across collections.
                """
                posterior = super().posterior_over_fuzzy_point(x, x_std)
                samples = posterior._tensor_sample_condensed(torch.Size((256,)))
                pred_samples = samples.exp()
                probs = (pred_samples / pred_samples.sum(TASK_DIM, keepdim=True)).mean(SAMPLE_DIM)
                detached_probs = probs.detach().cpu().numpy()
                predictions = detached_probs.argmax(axis=1)
                return predictions, detached_probs.max(axis=1)

            def _loss(self, train_x: torch.Tensor, train_y: torch.Tensor) -> torch.Tensor:
                """Accounting for multiple values."""
                return super()._loss(train_x, train_y).sum()

            @staticmethod
            def _noise_transform(gamma: numpy.typing.ArrayLike[float]) -> torch.Tensor:
                return torch.stack([torch.diag(torch.matmul(g, g.T)) for g in gamma], -1).squeeze().T

            @staticmethod
            def warn_normalise_y() -> None:
                """Override base warning because classification renders y normalisation irrelevant."""
                pass

        return InnerClass
