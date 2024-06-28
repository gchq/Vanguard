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
    def posterior_generator(num_failures_between_successes: int) -> Generator[Posterior, None, None]:
        """
        Yield invalid posteriors, with a valid posterior every `num_failures_between_successes`.

        Invalid posteriors have a negative definite covariance matrix.

        :param num_failures_between_successes: Number of invalid posteriors to yield between valid posteriors.
        """

        # These need to be actual classes and not just Mock instances to pass the type checking in
        # _create_updated_distribution().
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
                with self.assertRaisesRegex(RuntimeError, "errors in a row were caught while generating posteriors"):
                    MonteCarloPosteriorCollection(
                        posterior_generator=self.posterior_generator(
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
                # assertion: this doesn't raise an error
                collection = MonteCarloPosteriorCollection(
                    posterior_generator=self.posterior_generator(
                        num_failures_between_successes=self.failures_before_raise - 1
                    )
                )

            # check that we've had enough errors that we would have raised if they were in a row
            # pylint: disable-next=protected-access
            self.assertGreater(collection._posteriors_skipped, collection.MAX_POSTERIOR_ERRORS_BEFORE_RAISE)
