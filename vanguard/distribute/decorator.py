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
Contains the Distributed decorator.
"""

import warnings
from collections.abc import Iterable
from typing import Any, Generic, Optional, TypeVar, Union

import gpytorch
import numpy as np
import torch
from gpytorch.utils.warnings import GPInputWarning
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import override

from vanguard import utils
from vanguard.base import GPController
from vanguard.base.posteriors import Posterior
from vanguard.decoratorutils import TopMostDecorator, process_args, wraps_class
from vanguard.distribute.aggregators import (
    BadPriorVarShapeError,
    BaseAggregator,
    BCMAggregator,
    GRBCMAggregator,
    RBCMAggregator,
    XBCMAggregator,
    XGRBCMAggregator,
)
from vanguard.distribute.partitioners import BasePartitioner, KMeansPartitioner
from vanguard.features import HigherRankFeatures

_AGGREGATION_JITTER = 1e-10
_INPUT_WARNING = "The input matches the stored training data. Did you forget to call model.train()?"

ControllerT = TypeVar("ControllerT", bound=GPController)


class Distributed(TopMostDecorator, Generic[ControllerT]):
    """
    Use multiple controller classes to aggregate predictions.

    .. note::
        Because of the way expert controllers are created, the output standard deviation must be a
        float or an integer, and cannot be an array.

    .. note::
        Every call to :meth:`~vanguard.base.gpcontroller.GPController.fit` creates a new partition,
        and regenerates the experts.

    .. warning::
        This is a :class:`~vanguard.decoratorutils.basedecorator.TopMostDecorator`.

    :Example:
        >>> @Distributed(n_experts=10, aggregator_class=GRBCMAggregator)
        ... class DistributedGPController(GPController):
        ...     pass


    :param n_experts: The number of partitions in which to split the data. Defaults to 3.
    :param subset_fraction: The proportion of the training data to be used to train the hyperparameters.
        Defaults to 0.1.
    :param rng: Generator instance used to generate random numbers.
    :param aggregator_class: The class to be used for aggregation. Defaults to
        :class:`~vanguard.distribute.aggregators.RBCMAggregator`.
    :param partitioner_class: The class to be used for partitioning. Defaults to
        :class:`~vanguard.distribute.partitioners.KMeansPartitioner`. See
        :mod:`vanguard.distribute.partitioners` for alternative partitioners.
    :param partitioner_kwargs: Additional parameters passed to the partitioner initialisation.

    :Keyword Arguments:
        * For other possible keyword arguments, see the
          :class:`~vanguard.decoratorutils.basedecorator.Decorator` class.
    """

    def __init__(
        self,
        n_experts: int = 3,
        subset_fraction: float = 0.1,
        rng: Optional[np.random.Generator] = None,
        aggregator_class: type[BaseAggregator] = RBCMAggregator,
        partitioner_class: type[BasePartitioner] = KMeansPartitioner,
        partitioner_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the Distributed decorator.
        """
        self.n_experts = n_experts
        self.subset_fraction = subset_fraction
        self.rng = utils.optional_random_generator(rng)
        self.aggregator_class = aggregator_class
        self.partitioner_class = partitioner_class
        self.partitioner_kwargs = partitioner_kwargs if partitioner_kwargs is not None else {}
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    @override
    def verify_decorated_class(self, cls: type[ControllerT]) -> None:
        super().verify_decorated_class(cls)
        # pylint: disable-next=protected-access
        if HigherRankFeatures in cls.__decorators__ and not self.partitioner_class._can_handle_higher_rank_features:
            msg = (
                f"{self.partitioner_class.__name__} cannot handle higher-rank features. "
                "Consider moving the `@Distributed` decorator below the `@HigherRankFeatures` decorator."
            )
            raise TypeError(msg)

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
        from vanguard.hierarchical import LaplaceHierarchicalHyperparameters, VariationalHierarchicalHyperparameters
        from vanguard.learning import LearnYNoise
        from vanguard.multitask import Multitask
        from vanguard.normalise import NormaliseY
        from vanguard.standardise import DisableStandardScaling
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
                Classification: {
                    "fuzzy_predictive_likelihood",
                    "posterior_over_point",
                    "posterior_over_fuzzy_point",
                    "predictive_likelihood",
                },
                ClassificationMixin: {"classify_points", "classify_fuzzy_points"},
                DirichletKernelMulticlassClassification: {
                    "__init__",
                    "classify_points",
                    "classify_fuzzy_points",
                    "_get_predictions_from_prediction_means",
                },
                DirichletMulticlassClassification: {
                    "__init__",
                    "_loss",
                    "_noise_transform",
                    "classify_points",
                    "classify_fuzzy_points",
                    # "_get_predictions_from_prediction_means",
                    "warn_normalise_y",
                },
                DisableStandardScaling: {"_input_standardise_modules"},
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
                SetWarp: {"__init__", "_loss", "_sgd_round", "_unwarp_values", "warn_normalise_y"},
                SetInputWarp: {"__init__"},
                VariationalHierarchicalHyperparameters: {
                    "__init__",
                    "_fuzzy_predictive_likelihood",
                    "_get_posterior_over_fuzzy_point_in_eval_mode",
                    "_get_posterior_over_point",
                    "_gp_forward",
                    "_loss",
                    "_predictive_likelihood",
                },
                VariationalInference: {"_fuzzy_predictive_likelihood", "_predictive_likelihood", "__init__"},
            },
        )

    def _decorate_class(self, cls: type[ControllerT]) -> type[ControllerT]:
        decorator = self

        @wraps_class(cls, decorator_source=self)
        class InnerClass(cls):
            """
            Uses multiple controller classes to aggregate predictions.
            """

            _y_batch_axis = 0

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                self.rng = utils.optional_random_generator(all_parameters_as_kwargs.pop("rng", None))

                self._full_train_x = torch.as_tensor(all_parameters_as_kwargs.pop("train_x"))
                self._full_train_y = torch.as_tensor(all_parameters_as_kwargs.pop("train_y"))
                full_y_std = torch.as_tensor(all_parameters_as_kwargs.pop("y_std"))
                try:
                    self._full_y_std = full_y_std.item()
                except RuntimeError:
                    if full_y_std.ndim > 0:
                        raise TypeError(
                            f"The {type(self).__name__} class has been distributed, and can only accept a "
                            f"number or 0-dimensional array as the argument to 'y_std'. "
                            f"Got an array of shape {full_y_std.shape}."
                        ) from None
                    else:
                        raise

                self.aggregator_class = decorator.aggregator_class

                partitioner_class = decorator.partitioner_class
                partitioner_kwargs = dict(decorator.partitioner_kwargs)  # Copy so we don't change the original
                partitioner_kwargs.update(all_parameters_as_kwargs.pop("partitioner_kwargs", {}))
                communications_expert = issubclass(self.aggregator_class, (GRBCMAggregator, XGRBCMAggregator))
                self.partitioner = partitioner_class(
                    train_x=self._full_train_x,
                    n_experts=decorator.n_experts,
                    communication=communications_expert,
                    rng=self.rng,
                    **partitioner_kwargs,
                )

                self._expert_controllers: list[ControllerT] = []

                # pylint: disable=unbalanced-tuple-unpacking
                train_x_subset, train_y_subset, y_std_subset = _create_subset(
                    self._full_train_x,
                    self._full_train_y,
                    self._full_y_std,
                    subset_fraction=decorator.subset_fraction,
                    rng=decorator.rng,
                )

                self._expert_init_kwargs = all_parameters_as_kwargs
                super().__init__(
                    train_x=train_x_subset,
                    train_y=train_y_subset,
                    y_std=y_std_subset,
                    rng=self.rng,
                    **self._expert_init_kwargs,
                )

            def fit(self, n_sgd_iters: int = 10, gradient_every: int = 10) -> torch.Tensor:
                """
                Create the expert controllers.

                :param n_sgd_iters: The number of gradient updates to perform in each round of hyperparameter tuning.
                :param gradient_every: How often (in iterations) to do special HNIGP input gradient steps.
                    Defaults to same as `n_sgd_iters` normally, overridden to 1 in batch-mode.
                :returns: The loss.
                """
                loss = super().fit(n_sgd_iters, gradient_every=gradient_every)
                partition = self.partitioner.create_partition()
                self._expert_controllers = [
                    self._create_expert_controller(subset_indices) for subset_indices in partition
                ]
                return loss

            def expert_losses(self) -> list[float]:
                """
                Get the loss from each expert as evaluated on their subset of the data.

                .. warning::
                    This may not behave as expected on CUDA.

                :returns: The losses for each expert.
                """
                if self.device.type == "cuda":
                    warnings.warn("Collecting expert losses may not behave as expected on CUDA.", RuntimeWarning)

                losses = []
                for controller in self._expert_controllers:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=GPInputWarning, message=_INPUT_WARNING)
                        # pylint: disable=protected-access
                        loss = controller._loss(controller.train_x, controller.train_y)
                        losses.append(loss.detach().cpu().item())
                return losses

            def posterior_over_point(self, x: Union[NDArray[np.floating], torch.Tensor]) -> Posterior:
                """
                Aggregate expert posteriors.

                :param x: Data-point we wish to generate predictions for
                :return: Posterior for prediction of input `x`
                """
                expert_posteriors = (expert.posterior_over_point(x) for expert in self._expert_controllers)
                return self._aggregate_expert_posteriors(x, expert_posteriors)

            def posterior_over_fuzzy_point(
                self, x: Union[NDArray[np.floating], torch.Tensor], x_std: float
            ) -> Posterior:
                """
                Aggregate expert fuzzy posteriors.

                :param x: Data-point we wish to generate predictions for
                :param x_std: Standard deviation of noise associated with input `x`
                :return: Posterior for prediction of input `x`
                """
                expert_posteriors = (expert.posterior_over_fuzzy_point(x, x_std) for expert in self._expert_controllers)
                return self._aggregate_expert_posteriors(x, expert_posteriors)

            def _aggregate_expert_posteriors(
                self, x: Union[NDArray[np.floating], torch.Tensor], expert_posteriors: Iterable[Posterior]
            ) -> Posterior:
                """
                Aggregate an iterable of posteriors.

                :param x: The point at which the posteriors have been evaluated.
                :param expert_posteriors: The expert posteriors.
                :return: The aggregated posterior.
                """
                expert_distributions = (posterior.condensed_distribution for posterior in expert_posteriors)
                expert_means_and_covars = [
                    (distribution.mean, distribution.covariance_matrix) for distribution in expert_distributions
                ]
                aggregated_mean, aggregated_covar = self._aggregate_expert_predictions(x, expert_means_and_covars)
                aggregated_distribution = gpytorch.distributions.MultivariateNormal(aggregated_mean, aggregated_covar)
                aggregated_posterior = self.posterior_class(aggregated_distribution)
                return aggregated_posterior

            def _create_expert_controller(self, subset_indices: list[int]) -> ControllerT:
                """
                Create an expert controller with respect to a subset of the input data.

                :param subset_indices: List of indices to subset the training data using
                :return: Expert controller trained using a subset of the training data specified by `subset_indices`
                """
                train_x_subset, train_y_subset = self._full_train_x[subset_indices], self._full_train_y[subset_indices]
                try:
                    # TODO: _full_y_std is not allowed to be anything other than an int or float, so this will always
                    #  throw. Do we want to allow non-scalar values of _full_y_std? If not, delete this "try" block.
                    # https://github.com/gchq/Vanguard/issues/63
                    y_std_subset = self._full_y_std[subset_indices]
                except (TypeError, IndexError):
                    y_std_subset = self._full_y_std

                expert_controller = cls.new(
                    self, train_x=train_x_subset, train_y=train_y_subset, y_std=y_std_subset, **self._expert_init_kwargs
                )
                expert_controller.kernel.load_state_dict(self.kernel.state_dict())
                expert_controller.mean.load_state_dict(self.mean.state_dict())

                return expert_controller

            def _aggregate_expert_predictions(
                self,
                x: Union[NDArray[np.floating], NDArray[np.integer], torch.Tensor],
                means_and_covars: list[tuple[torch.Tensor, torch.Tensor]],
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """
                Aggregate the means and variances from the expert predictions.

                :param x: (n_predictions, n_features) The predictive inputs.
                :param means_and_covars: A list of (``mean``, ``variance``) pairs
                        representing the posterior predicted and mean for each expert controller.
                :returns: (``means``, ``covar``) where:

                    * ``means``: (n_predictions,) The posterior predictive mean,
                    * ``covar``: (n_predictions, n_predictions) The posterior predictive covariance.

                """
                prior_var = None
                if issubclass(self.aggregator_class, (BCMAggregator, RBCMAggregator, XBCMAggregator, XGRBCMAggregator)):
                    # diag=True is much faster than calling np.diag afterwards
                    prior_var = self.kernel(torch.as_tensor(x), diag=True).detach() + _AGGREGATION_JITTER

                means, covars = [], []
                for mean, covar in means_and_covars:
                    means.append(mean.detach())
                    covars.append(covar.detach())

                try:
                    aggregator = self.aggregator_class(means, covars, prior_var=prior_var)
                except BadPriorVarShapeError as exc:
                    raise RuntimeError(
                        "Cannot distribute using this kernel - try using a non-BCM aggregator instead."
                    ) from exc

                agg_mean, agg_covar = aggregator.aggregate()
                return agg_mean, agg_covar

        return InnerClass


def _create_subset(
    *arrays: Union[Tensor, NDArray[np.floating], NDArray[np.integer], float, int],
    subset_fraction: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> list[Union[Tensor, NDArray[np.floating], NDArray[np.integer], float]]:
    """
    Return subsets of the arrays along the same random indices.

    :param arrays: Subscriptable arrays. If an entry is not subscriptable it is returned as is
    :param subset_fraction: Fraction of points to include in the subset
    :param rng: Generator instance used to generate random numbers.
    :returns: The array subsets

    :Example:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([10, 20, 30, 40, 50])
        >>> z = 25
        >>>
        >>> _create_subset(x, y, subset_fraction=0.6, rng=np.random.default_rng(1))
        [array([3, 2, 4]), array([30, 20, 40])]
        >>> _create_subset(x, y, z, subset_fraction=0.6, rng=np.random.default_rng(1))
        [array([3, 2, 4]), array([30, 20, 40]), 25]
    """
    for array in arrays:
        try:
            length_of_first_subscriptable_array = array.shape[0]
            break
        except AttributeError:
            if isinstance(array, list):
                warnings.warn(
                    "Input 'arrays' are expected to be numpy arrays or floats. Got an array of type "
                    "`list' which will not be split into a subset."
                )
            continue
    else:
        # If the arrays contain no subscriptable arrays, just return them as a list
        return list(arrays)

    total_number_of_indices = length_of_first_subscriptable_array
    number_of_indices_in_subset = int(total_number_of_indices * subset_fraction)
    indices = rng.choice(total_number_of_indices, size=number_of_indices_in_subset, replace=False)

    subset_arrays = []
    for array in arrays:
        try:
            subset_array = array[indices]
        except (TypeError, IndexError):
            subset_array = array
        subset_arrays.append(subset_array)

    return subset_arrays
