"""
Contains the DirichletKernelMulticlassClassification decorator.
"""

from typing import Any, Tuple, Type, TypeVar, Union

import numpy as np
import numpy.typing
import torch

from vanguard import utils
from vanguard.base import GPController
from vanguard.classification.likelihoods import DirichletKernelClassifierLikelihood
from vanguard.classification.mixin import Classification, ClassificationMixin
from vanguard.classification.models import InertKernelModel
from vanguard.decoratorutils import Decorator, process_args, wraps_class

ControllerT = TypeVar("ControllerT", bound=GPController)
SAMPLE_DIM, TASK_DIM = 0, 2


class DirichletKernelMulticlassClassification(Decorator):
    """
    Implements multiclass classification using a Dirichlet kernel method.

    Based on the paper :cite:`MacKenzie14`.

    :Example:
        >>> from gpytorch.kernels import RBFKernel, ScaleKernel
        >>> import numpy as np
        >>> from vanguard.classification.likelihoods import (DirichletKernelClassifierLikelihood,
        ...                                                  GenericExactMarginalLogLikelihood)
        >>> from vanguard.vanilla import GaussianGPController
        >>>
        >>> @DirichletKernelMulticlassClassification(num_classes=3, ignore_methods=("__init__",))
        ... class MulticlassClassifier(GaussianGPController):
        ...     pass
        >>>
        >>> class Kernel(ScaleKernel):
        ...     def __init__(self) -> None:
        ...         super().__init__(RBFKernel())
        >>>
        >>> train_x = np.array([0, 0.1, 0.45, 0.55, 0.9, 1])
        >>> train_y = np.array([0, 0, 1, 1, 2, 2])
        >>>
        >>> gp = MulticlassClassifier(train_x, train_y, Kernel, y_std=0,
        ...                           likelihood_class=DirichletKernelClassifierLikelihood,
        ...                           marginal_log_likelihood_class=GenericExactMarginalLogLikelihood)
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
        :param kwargs: Keyword arguments passed to :class:`~vanguard.decoratorutils.basedecorator.Decorator`.
        """
        self.num_classes = num_classes
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        num_classes = self.num_classes

        @Classification()
        @wraps_class(cls)
        class InnerClass(cls, ClassificationMixin):
            gp_model_class = InertKernelModel

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                all_parameters_as_kwargs.pop("self")
                self.rng = utils.optional_random_generator(all_parameters_as_kwargs.pop("rng", None))

                likelihood_class = all_parameters_as_kwargs.pop("likelihood_class")
                if not issubclass(likelihood_class, DirichletKernelClassifierLikelihood):
                    raise ValueError(
                        "The class passed to `likelihood_class` must be a subclass of "
                        f"{DirichletKernelClassifierLikelihood.__name__}."
                    )

                train_y = all_parameters_as_kwargs.pop("train_y")

                likelihood_kwargs = all_parameters_as_kwargs.pop("likelihood_kwargs", {})
                model_kwargs = all_parameters_as_kwargs.pop("gp_kwargs", {})

                targets = torch.as_tensor(train_y, device=self.device, dtype=torch.int64)
                likelihood_kwargs["targets"] = targets
                likelihood_kwargs["num_classes"] = num_classes
                model_kwargs["num_classes"] = num_classes

                super().__init__(
                    train_y=train_y,
                    likelihood_class=likelihood_class,
                    likelihood_kwargs=likelihood_kwargs,
                    gp_kwargs=model_kwargs,
                    rng=self.rng,
                    **all_parameters_as_kwargs,
                )

            def classify_points(
                self, x: Union[float, numpy.typing.NDArray[np.floating]]
            ) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
                """Classify points."""
                means_as_floats, _ = super().predictive_likelihood(x).prediction()
                return self._get_predictions_from_prediction_means(means_as_floats)

            # TODO: throws an error - see linked issue
            # https://github.com/gchq/Vanguard/issues/288
            def classify_fuzzy_points(
                self,
                x: Union[float, numpy.typing.NDArray[np.floating]],
                x_std: Union[float, numpy.typing.NDArray[np.floating]],
            ) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
                """Classify fuzzy points."""
                means_as_floats, _ = super().fuzzy_predictive_likelihood(x, x_std).prediction()
                return self._get_predictions_from_prediction_means(means_as_floats)

            @staticmethod
            def _get_predictions_from_prediction_means(
                means: Union[float, numpy.typing.NDArray[np.floating]],
            ) -> Tuple[numpy.typing.NDArray[np.integer], numpy.typing.NDArray[np.floating]]:
                """
                Get the predictions and certainty probabilities from predictive likelihood means.

                :param means: The prediction means in the range [0, 1].
                :returns: The predicted class labels, and the certainty probabilities.
                """
                prediction = np.argmax(means, axis=1)
                certainty = np.max(means, axis=1)
                return prediction, certainty

        return InnerClass
