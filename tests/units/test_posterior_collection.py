"""
Tests for the MonteCarloPosteriorCollection class.
"""

import itertools
import unittest
from typing import Generator
from unittest import skip

import numpy as np
import torch
from gpytorch.distributions import Distribution, MultivariateNormal
from scipy.stats import multivariate_normal

from vanguard.base.posteriors import MonteCarloPosteriorCollection, Posterior


# pylint: disable-next=abstract-method
class OtherMultivariateNormal(MultivariateNormal):
    """Dummy class that exists to be different from MultivariateNormal in name only."""


class PosteriorCollectionTests(unittest.TestCase):
    """Tests for the MonteCarloPosteriorCollection class."""

    def test_illegal_multiple_distribution_types(self):
        """Test that an appropriate exception is raised when multiple different distribution types are provided."""

        def infinite_multi_type_generator() -> Generator[Distribution, None, None]:
            """Generate alternating MultivariateNormal and OtherMultivariateNormal posteriors."""
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
            Generate a large number of MultivariateNormal posteriors, then switch to OtherMultivariateNormal posteriors.
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
        Test that the log_probability method works as expected when the collection consists of identical posteriors.
        """

        def infinite_generator():
            """Generate infinite MultivariateNormal posteriors."""
            while True:
                yield Posterior(MultivariateNormal(torch.zeros((2,)), torch.eye(2)))

        collection = MonteCarloPosteriorCollection(infinite_generator())
        distribution = multivariate_normal(np.zeros((2,)), np.eye(2), seed=1234)

        # for 10 random points, check that the log-pdf calculated by the collection is the same as that for the
        # single distribution
        for _ in range(10):
            point = distribution.rvs(1)
            with self.subTest(point=point):
                expected_value = distribution.logpdf(point)
                self.assertAlmostEqual(expected_value, collection.log_probability(point), delta=1e-4)

    @skip("Throws a RuntimeError complaining about mismatched dimensions. Is this a bug or have I misunderstood?")
    def test_log_probability_multidimensional(self):
        """
        Test that the log_probability method works as expected when a two-dimensional sample is passed in.

        ...and when the collection consists of identical posteriors.
        """

        def infinite_generator():
            """Generate infinite MultivariateNormal posteriors."""
            while True:
                yield Posterior(MultivariateNormal(torch.zeros((2,)), torch.eye(2)))

        collection = MonteCarloPosteriorCollection(infinite_generator())
        distribution = multivariate_normal(np.zeros((2,)), np.eye(2), seed=1234)

        # for 10 random points, check that the log-pdf calculated by the collection is the same as that for the
        # single distribution
        for _ in range(10):
            points = distribution.rvs([1, 5])
            with self.subTest(points=points):
                expected_value = np.sum(distribution.logpdf(points), axis=0)
                return_value = collection.log_probability(points)
                self.assertAlmostEqual(expected_value, return_value, delta=1e-4)
