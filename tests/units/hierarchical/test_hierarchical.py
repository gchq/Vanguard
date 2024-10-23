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
Tests for the VariationalHierarchicalHyperparameters and BayesianHyperparameters decorators.
"""

import abc
import unittest
from typing import Any, Generic, TypeVar
from unittest.mock import MagicMock

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import RBFKernel, ScaleKernel

from tests.cases import get_default_rng
from vanguard.base import GPController
from vanguard.datasets.synthetic import MultidimensionalSyntheticDataset, SyntheticDataset
from vanguard.hierarchical import (
    BayesianHyperparameters,
    VariationalHierarchicalHyperparameters,
)
from vanguard.hierarchical.module import _descend_module_tree
from vanguard.vanilla import GaussianGPController

N_MC_SAMPLES = 13


@VariationalHierarchicalHyperparameters(num_mc_samples=N_MC_SAMPLES, ignore_all=True)
class VariationalFullBayesianGPController(GaussianGPController):
    pass


@BayesianHyperparameters()
class BayesianRBFKernel(RBFKernel):
    pass


class ScaledBayesianRBFKernel(ScaleKernel):
    def __init__(self, *args: Any, batch_shape: torch.Size = torch.Size([]), **kwargs: Any) -> None:
        super().__init__(BayesianRBFKernel(*args, batch_shape=batch_shape, **kwargs), batch_shape=batch_shape)


@BayesianHyperparameters()
class BayesianScaledRBFKernel(ScaleKernel):
    def __init__(self, *args: Any, batch_shape: torch.Size = torch.Size([]), **kwargs: Any) -> None:
        super().__init__(BayesianRBFKernel(*args, batch_shape=batch_shape, **kwargs), batch_shape=batch_shape)


class KernelConversionTests(unittest.TestCase):
    """
    Test functionality of kernels when handling Bayesian hyperparameters.
    """

    def test_kernel_bayesian_hyperparameters_prepared(self) -> None:
        """
        Verify the attributes of an RBF kernel that is decorated with BayesianHyperparameters are as expected.
        """
        kernel = BayesianRBFKernel()
        self.assertEqual(len(kernel.bayesian_hyperparameters), 1)
        self.assertEqual(kernel.bayesian_hyperparameters[0].raw_name, "raw_lengthscale")
        self.assertEqual(kernel.bayesian_hyperparameters[0].raw_shape, torch.Size([1, 1]))
        self.assertIsInstance(kernel.bayesian_hyperparameters[0].constraint, Positive)
        self.assertEqual(kernel.bayesian_hyperparameters[0].prior_mean, 8)
        self.assertEqual(kernel.bayesian_hyperparameters[0].prior_variance, 6**2)

    def test_kernel_bayesian_hyperparameters_prepared_ard_shape(self) -> None:
        """
        Verify that setting the number of dimensions works as expected for Bayesian hyperparameters.
        """
        kernel = BayesianRBFKernel(ard_num_dims=5)
        self.assertEqual(kernel.bayesian_hyperparameters[0].raw_shape, torch.Size([1, 5]))

    def test_kernel_bayesian_hyperparameters_variational_samples(self) -> None:
        """
        Verify that the lengthscale of the kernel is as expected when given data.

        We define a controller object using `VariationalFullBayesianGPController`, which has previously
        been defined to use a specific number of MCMC samples. We expect each hyperparameter of the supplied
        kernel to be a distribution with this many MCMC samples.
        """
        dataset = SyntheticDataset(rng=get_default_rng())
        gp = VariationalFullBayesianGPController(
            dataset.train_x,
            dataset.train_y,
            BayesianRBFKernel,
            dataset.train_y_std,
            rng=get_default_rng(),
        )
        self.assertEqual(gp.kernel.raw_lengthscale.shape, torch.Size([N_MC_SAMPLES, 1, 1]))

    def test_dimension_of_bayesian_variational_hyperparameters(self) -> None:
        """
        Verify that the collection of hyperparameters is as expected when given data.

        We define a controller object using `VariationalFullBayesianGPController`, which has previously
        been defined to use a specific number of MCMC samples. We expect each hyperparameter of the supplied
        kernel to be a distribution with this many MCMC samples.
        """
        dataset = SyntheticDataset(rng=get_default_rng())
        gp = VariationalFullBayesianGPController(
            dataset.train_x, dataset.train_y, ScaledBayesianRBFKernel, dataset.train_y_std, rng=get_default_rng()
        )
        self.assertEqual(gp.hyperparameter_collection.sample_tensor.shape, torch.Size([N_MC_SAMPLES, 1]))

    def test_predictive_likelihood(self) -> None:
        """
        Verify that the predictive likelihood is valid when using Bayesian hyperparameters.
        """
        dataset = SyntheticDataset(rng=get_default_rng())
        gp = VariationalFullBayesianGPController(
            dataset.train_x, dataset.train_y, ScaledBayesianRBFKernel, dataset.train_y_std, rng=get_default_rng()
        )
        gp.fit(10)

        # We are accessing a private method for testing, so we have to swap to eval mode and
        # then call the desired method
        gp.set_to_evaluation_mode()
        # pylint: disable=protected-access
        posterior = gp._predictive_likelihood(dataset.test_x)
        # pylint: enable=protected-access

        mean, upper, lower = posterior.confidence_interval()

        self.assertFalse(np.isnan(mean).any())
        self.assertFalse(np.isnan(upper).any())
        self.assertFalse(np.isnan(lower).any())

    def test_fuzzy_predictive_likelihood(self) -> None:
        """
        Verify that the fuzzy predictive likelihood is valid when using Bayesian hyperparameters.
        """
        dataset = SyntheticDataset(rng=get_default_rng())
        gp = VariationalFullBayesianGPController(
            dataset.train_x, dataset.train_y, ScaledBayesianRBFKernel, dataset.train_y_std, rng=get_default_rng()
        )
        gp.fit(10)

        # We are accessing a private method for testing, so we have to swap to eval mode and
        # then call the desired method
        gp.set_to_evaluation_mode()
        # pylint: disable=protected-access
        posterior = gp._fuzzy_predictive_likelihood(dataset.test_x, 0.05)
        # pylint: enable=protected-access

        mean, upper, lower = posterior.confidence_interval()

        self.assertFalse(np.isnan(mean).any())
        self.assertFalse(np.isnan(upper).any())
        self.assertFalse(np.isnan(lower).any())


GPControllerT = TypeVar("GPControllerT", bound=GPController)


class AbstractTests:
    """
    Test hierarchical functionality with an abstract controller.
    """

    # Namespace the test case ABCs below so they don't get run by unittest
    class TrainingTests(unittest.TestCase, Generic[GPControllerT], metaclass=abc.ABCMeta):
        """
        Basic tests for a hierarchical controller and BayesianHyperparameters decorators.
        """

        @property
        @abc.abstractmethod
        def controller_class(self) -> type[GPControllerT]:
            """
            The GPController subclass to be tested.

            Note that this isn't actually implemented as a property in subclasses of TrainingTest - it's only a property
            here so that we can type hint it without implementing it!
            """

        def test_non_bayesian_hyperparameters_are_point_estimates(self) -> None:
            """
            Verify the parameters of a Bayesian kernel are point estimates when not given a hierarchical controller.
            """
            dataset = SyntheticDataset()
            gp = self.controller_class(dataset.train_x, dataset.train_y, ScaledBayesianRBFKernel, dataset.train_y_std)
            gp.fit(10)
            self.assertNotEqual(gp.kernel.raw_outputscale.item(), 0.0)

        def test_posterior_does_not_fail(self) -> None:
            """
            Verify computing a posterior does not yield any nan values.

            We do not test that the posterior distribution reflects the data here, just that the posterior does not
            contain any nan values.
            """
            dataset = SyntheticDataset()
            gp = self.controller_class(dataset.train_x, dataset.train_y, ScaledBayesianRBFKernel, dataset.train_y_std)
            gp.fit(10)
            posterior = gp.posterior_over_point(dataset.test_x)
            mean, upper, lower = posterior.confidence_interval()
            self.assertNoNans(mean)
            self.assertNoNans(upper)
            self.assertNoNans(lower)

        def test_2d_non_bayesian_hyperparameters_are_point_estimates(self) -> None:
            """
            Verify that multidimensional hyperparameters are point estimates when not given a hierarchical controller.
            """
            dataset = MultidimensionalSyntheticDataset()
            gp = self.controller_class(
                dataset.train_x,
                dataset.train_y,
                ScaledBayesianRBFKernel,
                dataset.train_y_std,
                kernel_kwargs={"ard_num_dims": 2},
            )
            gp.fit(10)
            self.assertNotEqual(gp.kernel.raw_outputscale.item(), 0.0)

        def test_2d_posterior_does_not_fail(self) -> None:
            """
            Verify computing a multidimensional posterior does not yield any nan values.

            We do not test that the posterior distribution reflects the data here, just that the posterior does
            not contain any nan values.
            """
            dataset = MultidimensionalSyntheticDataset()
            gp = self.controller_class(
                dataset.train_x,
                dataset.train_y,
                ScaledBayesianRBFKernel,
                dataset.train_y_std,
                kernel_kwargs={"ard_num_dims": 2},
            )
            gp.fit(10)
            posterior = gp.posterior_over_point(dataset.test_x)
            mean, upper, lower = posterior.confidence_interval()
            self.assertNoNans(mean)
            self.assertNoNans(upper)
            self.assertNoNans(lower)

        def assertNoNans(self, array: np.typing.NDArray) -> None:  # pylint: disable=invalid-name
            """
            Check if a provided array contains any nan values.

            :param array: Array we will check for any nan values
            """
            self.assertFalse(np.isnan(array).any())


class TestBayesianHyperparameterCreation(unittest.TestCase):
    """
    Test creation of BayesianHyperparameter objects
    """

    def test_ignored_parameters(self) -> None:
        """
        Test if parameters are correctly ignored from Bayesian decoration.
        """

        @BayesianHyperparameters(ignored_parameters=["lengthscale"])
        class BayesianRBFKernelIgnored(RBFKernel):
            pass

        # Create two kernels - one with lengthscale ignored and one without
        kernel_without_ignore = BayesianRBFKernel()
        kernel_with_ignore = BayesianRBFKernelIgnored()

        # Check that the kernels have the expected bayesian hyperparameters given the ignore statement
        self.assertEqual(len(kernel_without_ignore.bayesian_hyperparameters), 1)
        self.assertEqual(kernel_without_ignore.bayesian_hyperparameters[0].raw_name, "raw_lengthscale")
        self.assertEqual(len(kernel_with_ignore.bayesian_hyperparameters), 0)

    def test_descend_module_tree(self) -> None:
        """
        Test descending a module tree when the tree has more than one level.
        """
        mock_object = MagicMock(gpytorch.Module)
        mock_object.a = MagicMock(gpytorch.Module)
        parameter_ancestry = ["a", "b"]

        result = _descend_module_tree(mock_object, parameter_ancestry)

        # Check the output - we expect to get mock_object.a as the first return, and 'b'
        # (the next parameter in the list) as the second return
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], mock_object.a)
        self.assertEqual(result[1], "b")
