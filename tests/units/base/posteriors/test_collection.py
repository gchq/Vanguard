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
Tests for the MonteCarloPosteriorCollection class.
"""

import itertools
import unittest
from typing import Generator
from unittest import expectedFailure
from unittest.mock import Mock, patch

import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal
from torch import Tensor
from torch.distributions import Distribution

from tests.cases import get_default_rng
from vanguard.base.posteriors import MonteCarloPosteriorCollection, Posterior


# pylint: disable-next=abstract-method
class OtherMultivariateNormal(MultivariateNormal):
    """Dummy class that exists to be different from `MultivariateNormal` in name only."""


class PosteriorCollectionTests(unittest.TestCase):
    """Tests for the `MonteCarloPosteriorCollection` class."""

    def test_illegal_multiple_distribution_types(self):
        """Test that an appropriate exception is raised when multiple different distribution types are provided."""

        def infinite_multi_type_generator() -> Generator[Distribution, None, None]:
            """Generate alternating `MultivariateNormal` and `OtherMultivariateNormal` posteriors."""
            for i in itertools.count():
                if i % 2 == 0:
                    yield Posterior(MultivariateNormal(torch.zeros((2,)), torch.eye(2)))
                else:
                    yield Posterior(OtherMultivariateNormal(torch.zeros((2,)), torch.eye(2)))

        with self.assertRaises(TypeError) as ctx:
            _ = MonteCarloPosteriorCollection(infinite_multi_type_generator())

        types = {MultivariateNormal, OtherMultivariateNormal}
        self.assertEqual(
            f"Posteriors have multiple distribution types: {types!r}.",
            str(ctx.exception),
        )

    def test_illegal_multiple_distribution_types_2(self):
        """
        Test that an appropriate exception is raised when multiple different distribution types are provided.

        The error raised here is different from the previous test as each sample only contains one distribution type,
        but subsequent samples contain different types.
        """

        def infinite_multi_type_generator() -> Generator[Distribution, None, None]:
            """
            Generate a large number of `MultivariateNormal` posteriors, then switch to `OtherMultivariateNormal`.
            """
            for i in itertools.count():
                if i < MonteCarloPosteriorCollection.INITIAL_NUMBER_OF_SAMPLES:
                    yield Posterior(MultivariateNormal(torch.zeros((2,)), torch.eye(2)))
                else:
                    yield Posterior(OtherMultivariateNormal(torch.zeros((2,)), torch.eye(2)))

        collection = MonteCarloPosteriorCollection(infinite_multi_type_generator())
        with self.assertRaises(TypeError) as ctx:
            collection.sample(1)

        self.assertEqual(
            f"Cannot add {OtherMultivariateNormal} types to {MultivariateNormal}.",
            str(ctx.exception),
        )

    def test_illegal_finite_generator(self):
        """Test that an appropriate exception is raised when a finite generator is given."""
        with self.assertRaises(RuntimeError) as ctx:
            _ = MonteCarloPosteriorCollection(
                p for p in [Posterior.from_mean_and_covariance(torch.zeros((2,)), torch.eye(2))]
            )
        self.assertEqual(
            "ran out of samples from the generator! "
            "MonteCarloPosteriorCollection must be given an infinite generator.",
            str(ctx.exception),
        )

    def test_log_probability(self):
        """
        Test that the `log_probability` method works as expected when the collection consists of identical posteriors.
        """

        def infinite_generator():
            """Generate infinite `MultivariateNormal` posteriors."""
            while True:
                yield Posterior(MultivariateNormal(torch.zeros((2,)), torch.eye(2)))

        collection = MonteCarloPosteriorCollection(infinite_generator())
        distribution = multivariate_normal(np.zeros((2,)), np.eye(2), seed=get_default_rng())

        # For 10 random points, check that the log-pdf calculated by the collection is the same as that for the
        # single distribution
        for _ in range(10):
            point = distribution.rvs(1)
            with self.subTest(point=point):
                expected_value = distribution.logpdf(point)
                self.assertAlmostEqual(expected_value, collection.log_probability(point), delta=1e-4)

    @expectedFailure
    # Throws a RuntimeError complaining about mismatched dimensions. TODO: This seems to be a bug.
    # https://github.com/gchq/Vanguard/issues/260

    def test_log_probability_multidimensional(self):
        """
        Test that the `log_probability` method works as expected when a two-dimensional sample is passed in.

        ...and when the collection consists of identical posteriors.
        """

        def infinite_generator():
            """Generate infinite `MultivariateNormal` posteriors."""
            while True:
                yield Posterior(MultivariateNormal(torch.zeros((2,)), torch.eye(2)))

        collection = MonteCarloPosteriorCollection(infinite_generator())
        distribution = multivariate_normal(np.zeros((2,)), np.eye(2), seed=get_default_rng())

        # For 10 random points, check that the log-pdf calculated by the collection is the same as that for the
        # single distribution
        for _ in range(10):
            points = distribution.rvs([1, 5])
            with self.subTest(points=points):
                expected_value = np.sum(distribution.logpdf(points), axis=0)
                return_value = collection.log_probability(points)
                self.assertAlmostEqual(expected_value, return_value, delta=1e-4)


class YieldPosteriorsTests(unittest.TestCase):
    """Tests for the _yield_posteriors method."""

    def setUp(self) -> None:
        """Set up data shared between tests."""
        self.failures_before_raise = 5

    @staticmethod
    def generate_mostly_invalid_posteriors(num_failures_between_successes: int) -> Generator[Posterior, None, None]:
        """
        Yield invalid posteriors, with a valid posterior every `num_failures_between_successes`.

        This is used to test how MonteCarloPosteriorCollection deals with invalid posteriors - internally,
        MonteCarloPosteriorCollection keeps track of how many invalid posteriors it's seen in a row, and raises an
        error if it sees too many *in a row*. This is to prevent it from getting stuck in an infinite loop if it gets a
        generator that - for whatever reason - *only* generates invalid posteriors.

        Thus, we generate invalid posteriors, with valid posteriors at regular intervals. If that interval is shorter
        than MonteCarloPosteriorCollection's threshold for "invalid posteriors in a row before raising an error", then
        we expect no error to be raised; if it is larger we expect to see an error.

        Invalid posteriors have a negative definite covariance matrix.

        Note also that this function returns manually-created mock classes rather than just `Mock` instances. This is
        because the type checking in `_create_updated_distribution()` uses `type()` to check that the distribution
        types match, and `type(Mock()) != type(Mock())`(!), so `Mock` instances can't pass the type checks.

        :param num_failures_between_successes: Number of invalid posteriors to yield between valid posteriors.
        """

        # These need to be actual classes rather than just Mock instances - see docstring for details
        class MockDistribution:
            """Mock of Distribution class."""

            __class__ = MultivariateNormal  # Lying to the type checker!

            def __init__(self, mean: torch.Tensor, covariance: torch.Tensor):
                self.mean = mean
                self.covariance_matrix = covariance

            def add_jitter(self, _):
                return self

            def rsample(self, *_, **__):
                return Mock(Tensor)

        class MockPosterior:
            """Mock of Posterior class."""

            def __init__(self, mean: torch.Tensor, covariance: torch.Tensor):
                self.distribution = MockDistribution(mean, covariance)

        failure_count = 0
        while True:
            if failure_count >= num_failures_between_successes:
                # Return a posterior with a positive definite covariance matrix.
                yield MockPosterior(torch.zeros(3), torch.eye(3))
                failure_count = 0
            else:
                # Return a posterior with a negative definite covariance matrix.
                yield MockPosterior(torch.zeros(3), -torch.eye(3))
                failure_count += 1

    def test_enough_posterior_errors_in_a_row_causes_raise(self):
        """Test that if enough errors are raised in a row, _yield_posteriors raises an error."""
        # Patch the number of errors before raising and initial number of samples to be much lower, for test speed.
        with patch.object(
            MonteCarloPosteriorCollection, "MAX_POSTERIOR_ERRORS_BEFORE_RAISE", self.failures_before_raise
        ):
            with patch.object(MonteCarloPosteriorCollection, "INITIAL_NUMBER_OF_SAMPLES", 10):
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"{self.failures_before_raise} errors in a row were caught while generating posteriors",
                ):
                    MonteCarloPosteriorCollection(
                        posterior_generator=self.generate_mostly_invalid_posteriors(
                            num_failures_between_successes=self.failures_before_raise + 1
                        )
                    )

    def test_posterior_errors_dont_cause_raise_if_they_are_not_in_a_row(self):
        """Test that even if many errors are raised, if they're not in a row, _yield_posteriors raises no errors."""
        # Patch the number of errors before raising and initial number of samples to be much lower, for test speed.
        with patch.object(
            MonteCarloPosteriorCollection, "MAX_POSTERIOR_ERRORS_BEFORE_RAISE", self.failures_before_raise
        ):
            with patch.object(MonteCarloPosteriorCollection, "INITIAL_NUMBER_OF_SAMPLES", 10):
                # This shouldn't raise an error, because the number of successive failures is below the threshold.
                # (This is the main thing that this test is testing.)
                collection = MonteCarloPosteriorCollection(
                    posterior_generator=self.generate_mostly_invalid_posteriors(
                        num_failures_between_successes=self.failures_before_raise - 1
                    )
                )

            # Check that we've had enough errors that we would have raised if they were in a row.
            # pylint: disable-next=protected-access
            self.assertGreater(collection._posteriors_skipped, collection.MAX_POSTERIOR_ERRORS_BEFORE_RAISE)
