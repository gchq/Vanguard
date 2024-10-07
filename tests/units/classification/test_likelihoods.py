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

"""Tests for DirichletKernelClassifierLikelihood."""

from unittest import TestCase, expectedFailure
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch.testing
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ZeroMean
from linear_operator import to_linear_operator
from torch.distributions import Distribution

from tests.cases import get_default_rng
from vanguard.classification.kernel import DirichletKernelMulticlassClassification
from vanguard.classification.likelihoods import (
    DirichletKernelClassifierLikelihood,
    GenericExactMarginalLogLikelihood,
    MultitaskBernoulliLikelihood,
)
from vanguard.classification.models import DummyKernelDistribution
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.vanilla import GaussianGPController


class TestDirichletKernelClassifierLikelihood(TestCase):
    """Tests for the `DirichletKernelClassifierLikelihood` class."""

    @classmethod
    def setUpClass(cls):
        """Set up immutable data shared between tests."""
        cls.num_classes = 4
        cls.dataset = MulticlassGaussianClassificationDataset(
            num_train_points=cls.num_classes * 3,
            num_test_points=cls.num_classes,
            num_classes=cls.num_classes,
            rng=get_default_rng(),
        )
        cls.likelihood = DirichletKernelClassifierLikelihood(num_classes=cls.num_classes)

    def setUp(self) -> None:
        """Set up data shared between tests."""
        self.rng = get_default_rng()

    @pytest.mark.no_beartype
    def test_illegal_input_type(self):
        """Test that we get an appropriate error when an illegal argument type is passed."""
        # Various illegal inputs
        illegal_inputs = [object(), np.array([1, 2, 3]), "string"]

        for illegal_input in illegal_inputs:
            with self.subTest(repr(illegal_input)):
                with self.assertRaises(TypeError) as ctx:
                    # Ignore type: it's intentionally incorrect
                    self.likelihood(illegal_input)  # type: ignore
                self.assertEqual(
                    "Likelihoods expects a DummyKernelDistribution input to make marginal predictions, or a "
                    f"torch.Tensor for conditional predictions. Got a {type(illegal_input).__name__}",
                    ctx.exception.args[0],
                )

    def test_alpha(self):
        """Test that when a value for `alpha` is provided to the initialiser, it's set correctly."""
        alpha = self.rng.uniform(2, 10)  # Ensuring alpha != 1
        likelihood = DirichletKernelClassifierLikelihood(num_classes=self.num_classes, alpha=alpha)
        torch.testing.assert_close(torch.ones(self.num_classes) * alpha, likelihood.alpha)

    def test_learn_alpha(self):
        """Test that when `learn_alpha` is True, the value of `alpha` is changed during fitting."""

        @DirichletKernelMulticlassClassification(num_classes=self.num_classes, ignore_methods=("__init__",))
        class MulticlassGaussianClassifier(GaussianGPController):
            """A simple Dirichlet multiclass classifier."""

        controller = MulticlassGaussianClassifier(
            train_x=self.dataset.train_x,
            train_y=self.dataset.train_y,
            y_std=0.0,
            mean_class=ZeroMean,
            kernel_class=RBFKernel,
            likelihood_class=DirichletKernelClassifierLikelihood,
            likelihood_kwargs={"learn_alpha": True, "alpha": 1.0},
            marginal_log_likelihood_class=GenericExactMarginalLogLikelihood,
            rng=self.rng,
        )

        starting_alpha = controller.likelihood.alpha.clone()
        controller.fit(1)
        fitted_alpha = controller.likelihood.alpha

        # Assert that alpha has changed
        assert not torch.all(torch.isclose(fitted_alpha, starting_alpha))

    def test_learn_alpha_constrained(self):
        """
        Test that when `learn_alpha` is True, and a constraint is supplied, that constraint is adhered to.
        """

        @DirichletKernelMulticlassClassification(num_classes=self.num_classes, ignore_methods=("__init__",))
        class MulticlassGaussianClassifier(GaussianGPController):
            """A simple Dirichlet multiclass classifier."""

        constraint_value = 0.5
        constrained_controller = MulticlassGaussianClassifier(
            train_x=self.dataset.train_x,
            train_y=self.dataset.train_y,
            y_std=0.0,
            mean_class=ZeroMean,
            kernel_class=RBFKernel,
            likelihood_class=DirichletKernelClassifierLikelihood,
            likelihood_kwargs={"learn_alpha": True, "alpha": 1.0, "alpha_constraint": GreaterThan(constraint_value)},
            marginal_log_likelihood_class=GenericExactMarginalLogLikelihood,
            rng=self.rng,
        )
        unconstrained_controller = MulticlassGaussianClassifier(
            train_x=self.dataset.train_x,
            train_y=self.dataset.train_y,
            y_std=0.0,
            mean_class=ZeroMean,
            kernel_class=RBFKernel,
            likelihood_class=DirichletKernelClassifierLikelihood,
            likelihood_kwargs={"learn_alpha": True, "alpha": 1.0},
            marginal_log_likelihood_class=GenericExactMarginalLogLikelihood,
            rng=self.rng,
        )

        constrained_controller.fit(10)
        unconstrained_controller.fit(10)
        constrained_alpha = constrained_controller.likelihood.alpha
        unconstrained_alpha = unconstrained_controller.likelihood.alpha

        constraint_limit = torch.ones_like(constrained_alpha) * constraint_value

        # Assert that when unconstrained, alpha drops below the constraint value
        assert torch.all(unconstrained_alpha < constraint_limit)
        # Assert that when constrained, alpha stays above the constraint value
        assert torch.all(constrained_alpha > constraint_limit)

    def test_log_marginal(self):
        """
        Test that `log_marginal` gives the log-probabilities of the marginal distribution.

        That is, we check that `log_marginal(x, dist)` == `marginal(dist).log_prob(x)`.
        """
        kernel = torch.tensor(self.rng.uniform(1, 2, size=(self.num_classes, self.num_classes)), dtype=torch.float)
        distribution = DummyKernelDistribution(
            to_linear_operator(torch.eye(self.num_classes, dtype=torch.float)), to_linear_operator(kernel)
        )
        observations = torch.tensor(self.rng.standard_normal(size=self.num_classes), dtype=torch.float)

        log_prob_direct = self.likelihood.log_marginal(observations, distribution)
        log_prob_indirect = self.likelihood.marginal(distribution).log_prob(observations)
        torch.testing.assert_close(log_prob_direct, log_prob_indirect)

    def test_call_conditional(self):
        """
        Test that when the likelihood is called directly with a tensor, the conditional distribution is returned.

        This is tested by just checking that the parent class (`Likelihood`) has its `__call__` method called.
        """
        input_tensor = torch.tensor(self.rng.standard_normal(size=1), dtype=torch.float)
        with patch.object(Likelihood, "__call__", return_value=Mock(Distribution)) as mock_super_call:
            self.likelihood(input_tensor)
        mock_super_call.assert_called_once_with(input_tensor)

    def test_call_marginal(self):
        """
        Test that when the likelihood is called with a `DummyKernelDistribution`, the marginal distribution is returned.

        This is tested by just checking that `marginal()` is called.
        """
        kernel = torch.tensor(self.rng.uniform(1, 2, size=(self.num_classes, self.num_classes)), dtype=torch.float)
        distribution = DummyKernelDistribution(
            to_linear_operator(torch.eye(self.num_classes, dtype=torch.float)), to_linear_operator(kernel)
        )
        with patch.object(
            DirichletKernelClassifierLikelihood, "marginal", return_value=Mock(Distribution)
        ) as mock_marginal:
            self.likelihood(distribution)
        mock_marginal.assert_called_once_with(distribution)


class TestMultitaskBernoulliLikelihood(TestCase):
    """Tests for the `MultitaskBernoulliLikelihood` class."""

    def setUp(self) -> None:
        """Set up data shared between tests."""
        self.rng = get_default_rng()

    # TODO: Fails with `AttributeError: 'super' object has no attribute 'log_prob'`.
    # https://github.com/gchq/Vanguard/issues/218
    @expectedFailure
    def test_log_marginal(self):
        """
        Test that `log_marginal` gives the log-probabilities of the marginal distribution.

        That is, we check that `log_marginal(x, dist)` == `marginal(dist).log_prob(x)`.
        """
        likelihood = MultitaskBernoulliLikelihood()
        size = 3
        mean = torch.tensor(self.rng.standard_normal(size=size), dtype=torch.float)
        distribution = MultivariateNormal(mean, torch.eye(size, dtype=torch.float))
        observations = torch.tensor(self.rng.standard_normal(size=5), dtype=torch.float)

        log_prob_direct = likelihood.log_marginal(observations, distribution)
        log_prob_indirect = likelihood.marginal(distribution).log_prob(observations)
        torch.testing.assert_close(log_prob_direct, log_prob_indirect)
