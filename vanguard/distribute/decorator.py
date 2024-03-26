"""
Contains the Distributed decorator.
"""
import warnings
from typing import TypeVar, Generic, Iterable, Optional, Union

import gpytorch
from gpytorch.utils.warnings import GPInputWarning
import numpy as np
import torch
from numpy.typing import NDArray

from ..base import GPController
from ..base.posteriors import Posterior
from ..decoratorutils import TopMostDecorator, process_args, wraps_class
from .aggregators import (BadPriorVarShapeError, BCMAggregator, GRBCMAggregator, RBCMAggregator, XBCMAggregator,
                          XGRBCMAggregator, BaseAggregator)
from .partitioners import KMeansPartitioner, KMedoidsPartitioner, BasePartitioner

_AGGREGATION_JITTER = 1e-10
_INPUT_WARNING = "The input matches the stored training data. Did you forget to call model.train()?"

ControllerT = TypeVar("ControllerT", bound=GPController)
class Distributed(TopMostDecorator, Generic[ControllerT]):
    """
    Uses multiple controller classes to aggregate predictions.

    .. note::
        Because of the way expert controllers are created, the output standard deviation must be a
        float or an integer, and cannot be an array.

    .. note::
        Every call to :meth:`~vanguard.base.gpcontroller.GPController.fit` creates a new partition,
        and regenerates the experts.

    :Example:
        >>> @Distributed(n_experts=10, aggregator_class=GRBCMAggregator)
        ... class DistributedGPController(GPController):
        ...     pass
    """
    def __init__(self,
                 n_experts: int = 3,
                 subset_fraction: float = 0.1,
                 seed: Optional[int] = 42,
                 aggregator_class: type[BaseAggregator] = RBCMAggregator,
                 partitioner_class: type[BasePartitioner] = KMeansPartitioner,
                 **kwargs):
        """
        Initialise self.

        :param int n_experts: The number of partitions in which to split the data. Defaults to 3.
        :param float subset_fraction: The proportion of the training data to be used to train the hyperparameters.
            Defaults to 0.1.
        :param seed: The seed used for creating the subset of the training data used to train the hyperparameters.
            Defaults to 42.
        :param type aggregator_class: The class to be used for aggregation. Defaults to
            class:`~vanguard.distribute.aggregators.RBCMAggregator`.
        :param type partitioner_class: The class to be used for partitioning. Defaults to
            class:`~vanguard.distribute.partitioners.KMeansPartitioner`.

        :Keyword Arguments:

            * **partitioner_args** *dict*: Additional parameters passed to the partitioner initialisation.
            * For other possible keyword arguments, see the
              class:`~vanguard.decoratorutils.basedecorator.Decorator` class.
        """
        self.n_experts = n_experts
        self.subset_fraction = subset_fraction
        self.seed = seed
        self.aggregator_class = aggregator_class
        self.partitioner_class = partitioner_class
        self.partitioner_kwargs = kwargs.pop("partitioner_kwargs", {})
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: type[ControllerT]) -> type[ControllerT]:
        decorator = self

        @wraps_class(cls)
        class InnerClass(cls):
            """
            Uses multiple controller classes to aggregate predictions.
            """
            _y_batch_axis = 0

            def __init__(self, *args, **kwargs):
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                all_parameters_as_kwargs.pop("self")

                self._full_train_x: NDArray = all_parameters_as_kwargs.pop("train_x")
                self._full_train_y: NDArray = all_parameters_as_kwargs.pop("train_y")
                self._full_y_std: Union[int, float] = all_parameters_as_kwargs.pop("y_std")

                if not isinstance(self._full_y_std, (float, int)):
                    raise TypeError(f"The {type(self).__name__} class has been distributed, and can only accept a "
                                    f"number as the argument to 'y_std', not '{type(self._full_y_std).__name__}'.")

                self.aggregator_class = decorator.aggregator_class

                partitioner_class = decorator.partitioner_class
                partitioner_kwargs = decorator.partitioner_kwargs
                if issubclass(partitioner_class, KMedoidsPartitioner):
                    partitioner_kwargs["kernel"] = all_parameters_as_kwargs["kernel"]
                communications_expert = issubclass(self.aggregator_class, (GRBCMAggregator, XGRBCMAggregator))
                self.partitioner = partitioner_class(train_x=self._full_train_x, n_experts=decorator.n_experts,
                                                     communication=communications_expert, **partitioner_kwargs)

                self._expert_controllers: list[ControllerT] = []

                train_x_subset, train_y_subset, y_std_subset = _create_subset(self._full_train_x,
                                                                              self._full_train_y,
                                                                              self._full_y_std,
                                                                              subset_fraction=decorator.subset_fraction,
                                                                              seed=decorator.seed)

                self._expert_init_kwargs = all_parameters_as_kwargs
                super().__init__(train_x=train_x_subset, train_y=train_y_subset, y_std=y_std_subset,
                                 **self._expert_init_kwargs)

            def fit(self, n_sgd_iters: int = 10, gradient_every: int = 10) -> torch.Tensor:
                """Also create the expert controllers."""
                loss = super().fit(n_sgd_iters, gradient_every=gradient_every)
                partition = self.partitioner.create_partition()
                self._expert_controllers = [self._create_expert_controller(subset_indices)
                                            for subset_indices in partition]
                return loss

            def expert_losses(self) -> list[float]:
                """
                Get the loss from each expert as evaluated on their subset of the data.

                .. warning::
                    This may not behave as expected on CUDA.

                :returns: The losses for each expert.
                :rtype: list[float]
                """
                if self.device.type == "cuda":
                    warnings.warn("Collecting expert losses may not behave as expected on CUDA.", RuntimeWarning)

                losses = []
                for controller in self._expert_controllers:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=GPInputWarning, message=_INPUT_WARNING)
                        loss = controller._loss(controller.train_x, controller.train_y)
                        losses.append(loss.detach().cpu().item())
                return losses

            def posterior_over_point(self, x: Union[NDArray[np.floating], torch.Tensor]) -> Posterior:
                """Aggregate expert posteriors."""
                expert_posteriors = (expert.posterior_over_point(x) for expert in self._expert_controllers)
                return self._aggregate_expert_posteriors(x, expert_posteriors)

            def posterior_over_fuzzy_point(
                    self, x: Union[NDArray[np.floating], torch.Tensor], x_std: float
            ) -> Posterior:
                """Aggregate expert fuzzy posteriors."""
                expert_posteriors = (expert.posterior_over_fuzzy_point(x, x_std) for expert in self._expert_controllers)
                return self._aggregate_expert_posteriors(x, expert_posteriors)

            def _aggregate_expert_posteriors(self,
                                             x: Union[NDArray[np.floating], torch.Tensor],
                                             expert_posteriors: Iterable[Posterior]
                                             ) -> Posterior:
                """
                Aggregate an iterable of posteriors.

                :param torch.Tensor x: The point at which the posteriors have been evaluated.
                :param Iterable[vanguard.base.posteriors.Posterior] expert_posteriors: The expert posteriors.
                :return: The aggregated posterior.
                :rtype: vanguard.base.posteriors.Posterior
                """
                expert_distributions = (posterior.condensed_distribution for posterior in expert_posteriors)
                expert_means_and_covars = [(distribution.mean, distribution.covariance_matrix)
                                           for distribution in expert_distributions]
                aggregated_mean, aggregated_covar = self._aggregate_expert_predictions(x, expert_means_and_covars)
                aggregated_distribution = gpytorch.distributions.MultivariateNormal(aggregated_mean, aggregated_covar)
                aggregated_posterior = self.posterior_class(aggregated_distribution)
                return aggregated_posterior

            def _create_expert_controller(self, subset_indices) -> ControllerT:
                """Create an expert controller with respect to a subset of the input data."""
                train_x_subset, train_y_subset = self._full_train_x[subset_indices], self._full_train_y[subset_indices]
                try:
                    # TODO: note to reviewer - why is this here? full_y_std is not allowed to be anything other than
                    #  an int or float
                    y_std_subset = self._full_y_std[subset_indices]
                except (TypeError, IndexError):
                    y_std_subset = self._full_y_std

                expect_controller = cls.new(self, train_x=train_x_subset, train_y=train_y_subset, y_std=y_std_subset)
                expect_controller.kernel.load_state_dict(self.kernel.state_dict())
                expect_controller.mean.load_state_dict(self.mean.state_dict())

                return expect_controller

            def _aggregate_expert_predictions(self,
                                              x: Union[NDArray[np.floating], torch.Tensor],
                                              means_and_covars: list[tuple[torch.Tensor, torch.Tensor]]
                                              ) -> tuple[torch.Tensor, torch.Tensor]:
                """
                Aggregate the means and variances from the expert predictions.

                :param array_like[float] x: (n_preds, n_features) The predictive inputs.
                :param list[tuple[Tensor[float]]] means_and_covars: A list of (``mean``, ``variance``) pairs
                        representing the posterior predicted and mean for each expert controller.
                :returns: (``means``, ``covar``) where:

                    * ``means``: (n_preds,) The posterior predictive mean,
                    * ``covar``: (n_preds, n_preds) The posterior predictive covariance.

                :rtype: tuple[torch.Tensor]
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
                except BadPriorVarShapeError:
                    raise RuntimeError("Cannot distribute using this kernel - try using a non-BCM aggregator instead.")

                agg_mean, agg_covar = aggregator.aggregate()
                return agg_mean, agg_covar

        return InnerClass


def _create_subset(*arrays: Union[NDArray[np.floating], float],
                   subset_fraction: float = 0.1,
                   seed: Optional[int] = None
                   ) -> list[Union[NDArray[np.floating], float]]:
    """
    Return subsets of the arrays along the same random indices.

    :param numpy.ndarray arrays: Subscriptable arrays. If an entry is not subscriptable it is returned as is.
    :returns: The subsetted arrays.
    :rtype: list[array_like,Any]

    :Example:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([10, 20, 30, 40, 50])
        >>> z = 25
        >>>
        >>> _create_subset(x, y, subset_fraction=0.6, seed=1)
        [array([3, 2, 5]), array([30, 20, 50])]
        >>> _create_subset(x, y, z, subset_fraction=0.6, seed=1)
        [array([3, 2, 5]), array([30, 20, 50]), 25]
    """
    for array in arrays:
        try:
            length_of_first_subscriptable_array = array.shape[0]
            break
        except AttributeError:
            continue
    else:
        return list(arrays)  # contains no subscriptable arrays

    np.random.seed(seed)
    total_number_of_indices = length_of_first_subscriptable_array
    number_of_indices_in_subset = int(total_number_of_indices * subset_fraction)
    indices = np.random.choice(total_number_of_indices, size=number_of_indices_in_subset, replace=False)

    subset_arrays = []
    for array in arrays:
        try:
            subset_array = array[indices]
        except (TypeError, IndexError):
            subset_array = array
        subset_arrays.append(subset_array)

    return subset_arrays
