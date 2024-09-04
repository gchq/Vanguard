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
Contains the HyperparameterCollection class.
"""

from typing import Any, Iterator, List, Tuple, Type, TypeVar

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal

from vanguard.hierarchical.hyperparameter import BayesianHyperparameter

HyperparameterT = TypeVar("HyperparameterT", bound=BayesianHyperparameter)
ModuleT = TypeVar("ModuleT", bound=gpytorch.module.Module)
VariationalDistributionT = TypeVar(
    "VariationalDistributionT",
    bound=gpytorch.variational._VariationalDistribution,  # pylint: disable=protected-access
)


class HyperparameterCollection:
    """
    Represents a collection of hyperparameters for a controller.

    This class will delete the original torch parameters for the hyperparameters
    so that they can be replaced by batches of parameters representing samples
    from a distribution over those hyperparameters.
    """

    def __init__(
        self,
        module_hyperparameter_pairs: List[Tuple[ModuleT, HyperparameterT]],
        sample_shape: torch.Size,
        variational_distribution_class: Type[VariationalDistributionT],
    ) -> None:
        """
        Initialise self.

        :param module_hyperparameter_pairs: A list of (module, hyperparameter) pairs.
        :param sample_shape: The shape of the sample tensor.
        :param variational_distribution_class: The variational
            distribution to use for the raw hyperparameters' posterior.
        """
        self.sample_shape = sample_shape
        self.module_hyperparameter_pairs = module_hyperparameter_pairs

        self.variational_dimension = sum(
            self._parameter_index_size(hyperparameter) for _, hyperparameter in module_hyperparameter_pairs
        )
        self.variational_distribution = variational_distribution_class(self.variational_dimension)

        self.prior_mean = torch.zeros(self.variational_dimension)
        self.prior_variance = torch.ones(self.variational_dimension)

        prior_covariance_matrix = torch.diag(self.prior_variance)
        self._inverse_prior_covariance_matrix = torch.diag(1 / self.prior_variance)
        self.prior = MultivariateNormal(self.prior_mean, prior_covariance_matrix)

        self.sample_tensor = None
        self._hyperparameter_to_index = {}

        self._delete_point_estimate_hyperparameters()
        self._sample()
        self._initialise_variational_parameters_and_constants()

        self.prior_mean.requires_grad = False
        self.prior_variance.requires_grad = False

    def sample_and_update(self) -> None:
        """Sample from the collection, and update the hyperparameters."""
        self._sample()
        for owner_module, hyperparameter in self.module_hyperparameter_pairs:
            self._update_hyperparameter_value(owner_module, hyperparameter)

    def kl_term(self) -> torch.Tensor:
        """Compute the KL divergence term in the ELBO."""
        mu = self.variational_distribution.variational_mean
        sigma = self.variational_distribution().covariance_matrix

        mu_0 = self.prior.mean
        sigma_0 = self.prior.covariance_matrix

        sigma_0_inv = self._inverse_prior_covariance_matrix
        trace_term = torch.trace(sigma_0_inv @ sigma)
        mean_diff = mu_0 - mu
        mean_term = mean_diff.t() @ sigma_0_inv @ mean_diff
        det_term = torch.log(torch.linalg.det(sigma_0) / torch.linalg.det(sigma))  # pylint: disable=not-callable

        return (trace_term + mean_term + det_term - mu.shape[0]) / 2

    def _sample(self) -> None:
        """Sample from the collection."""
        distribution = self.variational_distribution()
        self.sample_tensor = distribution.rsample(self.sample_shape)

    def _initialise_variational_parameters_and_constants(self) -> None:
        """Infer an index into the sample tensor for each hyperparameter, and initialise accordingly."""
        variational_index = 0

        for owner_module, hyperparameter in self.module_hyperparameter_pairs:
            index_size = self._parameter_index_size(hyperparameter)

            index = (slice(None), slice(variational_index, variational_index + index_size))
            self._hyperparameter_to_index[(owner_module, hyperparameter.raw_name)] = index

            self._update_hyperparameter_value(owner_module, hyperparameter)

            mean_var_slice = slice(variational_index, variational_index + index_size)

            self.prior_mean[mean_var_slice] = hyperparameter.prior_mean
            self.prior_variance[mean_var_slice] = hyperparameter.prior_variance

            variational_index += index_size

    def _parameter_index_size(self, hyperparameter: HyperparameterT) -> int:
        """
        Get the size of the index into the sample tensor corresponding to the hyperparameter.

        In order to ensure that all hyperparameters fit into the sample tensor (whose size can vary),
        this method will scale the size of the hyperparameter in order to return the correct
        proportional index size.
        """
        return hyperparameter.numel() // self.sample_shape[0]

    def _update_hyperparameter_value(self, owner_module: ModuleT, hyperparameter: HyperparameterT) -> None:
        """Update the value of a hyperparameter within its owner module."""
        index = self._hyperparameter_to_index[(owner_module, hyperparameter.raw_name)]
        sliced_tensor = self.sample_tensor[index].reshape(hyperparameter.raw_shape)
        setattr(owner_module, hyperparameter.raw_name, sliced_tensor)

    def _delete_point_estimate_hyperparameters(self) -> None:
        for owner_module, hyperparameter in self.module_hyperparameter_pairs:
            try:
                delattr(owner_module, hyperparameter.raw_name)
            except AttributeError:
                continue


class OnePointHyperparameterCollection:
    """
    Represents a collection of hyperparameters for a controller.

    This class keeps hyperparameters in their original shape and just manages
    the representation of the hyperparameters as a single combined tensor.
    It also manages the prior placed over the hyperparameters.
    """

    def __init__(self, module_hyperparameter_pairs: List[Tuple[ModuleT, HyperparameterT]]) -> None:
        """
        Initialise self.

        :param module_hyperparameter_pairs: A list of (module, hyperparameter) pairs.
        """
        self.module_hyperparameter_pairs = module_hyperparameter_pairs

        self.hyperparameter_dimension = sum(
            self._parameter_index_size(hyperparameter) for _, hyperparameter in module_hyperparameter_pairs
        )

        self.prior_mean = torch.zeros(self.hyperparameter_dimension)
        self.prior_variance = torch.ones(self.hyperparameter_dimension)

        self._hyperparameter_to_index = {}

        self._initialise_hyperparameter_indices()

        self.prior_mean.requires_grad = False
        self.prior_variance.requires_grad = False

        prior_covariance_matrix = torch.diag(self.prior_variance)
        self.prior = MultivariateNormal(self.prior_mean, prior_covariance_matrix)
        self.log_partition_function = self.prior.log_prob(self.prior_mean)

    def __iter__(self) -> Iterator[Any]:
        return (getattr(module, hyperparameter.raw_name) for module, hyperparameter in self.module_hyperparameter_pairs)

    def __len__(self) -> int:
        return len(self.module_hyperparameter_pairs)

    @property
    def hyperparameter_tensor(self) -> torch.Tensor:
        tensor = torch.zeros(self.hyperparameter_dimension)
        for owner_module, hyperparameter in self.module_hyperparameter_pairs:
            index = self._hyperparameter_to_index[(owner_module, hyperparameter.raw_name)]
            tensor[index] = getattr(owner_module, hyperparameter.raw_name)
        return tensor

    @hyperparameter_tensor.setter
    def hyperparameter_tensor(self, value: torch.Tensor) -> None:
        for owner_module, hyperparameter in self.module_hyperparameter_pairs:
            index = self._hyperparameter_to_index[(owner_module, hyperparameter.raw_name)]
            shape = getattr(owner_module, hyperparameter.raw_name).shape
            try:
                setattr(owner_module, hyperparameter.raw_name, value[index].reshape(shape))
            except TypeError:
                delattr(owner_module, hyperparameter.raw_name)
                setattr(owner_module, hyperparameter.raw_name, value[index].reshape(shape))

    def log_prior_term(self) -> torch.Tensor:
        """
        Compute the log un-normalised prior density.

        The partition function has in principle no effect on the optimisation
        but can skew the loss values unhelpfully, so we remove it.
        """
        return self.prior.log_prob(self.hyperparameter_tensor) - self.log_partition_function

    def _initialise_hyperparameter_indices(self) -> None:
        """Infer an index into the sample tensor for each hyperparameter, and initialise accordingly."""
        variational_index = 0

        for owner_module, hyperparameter in self.module_hyperparameter_pairs:
            index_size = self._parameter_index_size(hyperparameter)

            index = slice(variational_index, variational_index + index_size)
            self._hyperparameter_to_index[(owner_module, hyperparameter.raw_name)] = index

            mean_var_slice = slice(variational_index, variational_index + index_size)

            self.prior_mean[mean_var_slice] = hyperparameter.prior_mean
            self.prior_variance[mean_var_slice] = hyperparameter.prior_variance

            variational_index += index_size

    def _parameter_index_size(self, hyperparameter: HyperparameterT) -> int:
        """
        Get the size of the index into the sample tensor corresponding to the hyperparameter.

        In order to ensure that all hyperparameters fit into the sample tensor (whose size can vary),
        this method will scale the size of the hyperparameter in order to return the correct
        proportional index size.
        """
        return hyperparameter.numel()
