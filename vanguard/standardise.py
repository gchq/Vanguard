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
from vanguard.decoratorutils import Decorator, wraps_class

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
        # pylint: disable=import-outside-toplevel
        from vanguard.classification import (
            BinaryClassification,
            CategoricalClassification,
            DirichletMulticlassClassification,
        )
        from vanguard.classification.kernel import DirichletKernelMulticlassClassification
        from vanguard.classification.mixin import Classification, ClassificationMixin
        from vanguard.features import HigherRankFeatures
        from vanguard.hierarchical import LaplaceHierarchicalHyperparameters, VariationalHierarchicalHyperparameters
        from vanguard.learning import LearnYNoise
        from vanguard.multitask import Multitask
        from vanguard.normalise import NormaliseY
        from vanguard.variational import VariationalInference
        from vanguard.warps import SetInputWarp, SetWarp
        # pylint: enable=import-outside-toplevel

        return self._add_to_safe_updates(
            super().safe_updates,
            {
                BinaryClassification: {
                    "__init__",
                    "classify_points",
                    "classify_fuzzy_points",
                    "_get_predictions_from_prediction_means",
                    "warn_normalise_y",
                },
                CategoricalClassification: {
                    "__init__",
                    "classify_points",
                    "classify_fuzzy_points",
                    "_get_predictions_from_posterior",
                    "warn_normalise_y",
                },
                ClassificationMixin: {"classify_points", "classify_fuzzy_points"},
                Classification: {
                    "posterior_over_point",
                    "posterior_over_fuzzy_point",
                    "fuzzy_predictive_likelihood",
                    "predictive_likelihood",
                },
                DirichletMulticlassClassification: {
                    "__init__",
                    "_loss",
                    "_noise_transform",
                    "classify_points",
                    "classify_fuzzy_points",
                    "_get_predictions_from_prediction_means",
                    "warn_normalise_y",
                },
                DirichletKernelMulticlassClassification: {
                    "__init__",
                    "classify_points",
                    "classify_fuzzy_points",
                    "_get_predictions_from_prediction_means",
                },
                DisableStandardScaling: {"_input_standardise_modules"},
                HigherRankFeatures: {"__init__"},
                LaplaceHierarchicalHyperparameters: {
                    "__init__",
                    "_compute_hyperparameter_laplace_approximation",
                    "_compute_loss_hessian",
                    "_fuzzy_predictive_likelihood",
                    "_get_posterior_over_fuzzy_point_in_eval_mode",
                    "_get_posterior_over_point",
                    "_gp_forward",
                    "_predictive_likelihood",
                    "_sample_and_set_hyperparameters",
                    "_sgd_round",
                    "_update_hyperparameter_posterior",
                    "auto_temperature",
                },
                LearnYNoise: {"__init__"},
                Multitask: {"__init__", "_match_mean_shape_to_kernel"},
                NormaliseY: {"__init__", "warn_normalise_y"},
                SetInputWarp: {"__init__"},
                SetWarp: {"__init__", "_loss", "_sgd_round", "warn_normalise_y", "_unwarp_values"},
                VariationalHierarchicalHyperparameters: {
                    "__init__",
                    "_fuzzy_predictive_likelihood",
                    "_get_posterior_over_fuzzy_point_in_eval_mode",
                    "_get_posterior_over_point",
                    "_gp_forward",
                    "_loss",
                    "_predictive_likelihood",
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
