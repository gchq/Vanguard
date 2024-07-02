"""
Tests for the MonteCarloPosteriorCollection class.
"""

import unittest
from typing import Generator
from unittest.mock import Mock, patch

import torch

from vanguard.base.posteriors import MonteCarloPosteriorCollection, Posterior


class YieldPosteriorsTests(unittest.TestCase):
    """Tests for the _yield_posteriors method."""

    def setUp(self):
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

            def __init__(self, mean: torch.Tensor, covariance: torch.Tensor):
                self.mean = mean
                self.covariance_matrix = covariance

            def __getattr__(self, _):
                return Mock()

        class MockPosterior:
            """Mock of Posterior class."""

            def __init__(self, mean: torch.Tensor, covariance: torch.Tensor):
                self.distribution = MockDistribution(mean, covariance)

        failure_count = 0
        while True:
            if failure_count >= num_failures_between_successes:
                # return a posterior with a positive definite covariance matrix
                yield MockPosterior(torch.zeros(3), torch.eye(3))
                failure_count = 0
            else:
                # return a posterior with a negative definite covariance matrix
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

            # check that we've had enough errors that we would have raised if they were in a row
            # pylint: disable-next=protected-access
            self.assertGreater(collection._posteriors_skipped, collection.MAX_POSTERIOR_ERRORS_BEFORE_RAISE)
