"""
Tests for partitioner classes.
"""
import unittest

import numpy as np
from gpytorch.kernels import RBFKernel

from vanguard.distribute import partitioners


class PartitionTests(unittest.TestCase):
    """
    Test that the results are the same.

    Note that these tests are for reproducibility, and assume that the results
    of the functions were originally correct.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        rng = np.random.RandomState(seed=1)
        self.train_x = rng.random(size=10).reshape(-1, 1) * 20
        self.kernel = RBFKernel()
        self.n_experts = 3
        self.seed = 42

        self.expected_partition_results = {
            partitioners.RandomPartitioner: [[8, 1, 5], [0, 7, 2], [9, 4, 3]],
            partitioners.KMeansPartitioner: [[2, 4, 5, 6], [1, 9], [0, 3, 7, 8]],
            partitioners.MiniBatchKMeansPartitioner: [[0, 3, 7, 8], [1, 9], [2, 4, 5, 6]],
            partitioners.KMedoidsPartitioner: [[1, 3, 7], [2, 4, 5, 6], [0, 8, 9]],
        }

        self.expected_communication_partition_results = {
            partitioners.RandomPartitioner: [[8, 1, 5], [8, 1, 5, 0, 1, 8, 5, 3], [8, 1, 5, 4, 7, 9, 6, 2]],
            partitioners.KMeansPartitioner: [[8, 1, 5], [8, 1, 5, 0, 1, 3, 7, 8, 9], [8, 1, 5, 2, 4, 5, 6]],
            partitioners.MiniBatchKMeansPartitioner: [[8, 1, 5], [8, 1, 5, 2, 4, 5, 6], [8, 1, 5, 0, 1, 3, 7, 8, 9]],
            partitioners.KMedoidsPartitioner: [[8, 1, 5], [8, 1, 5, 0, 1, 3, 7, 8, 9], [8, 1, 5, 2, 4, 5, 6]],
        }

    @unittest.skip("Fails on 3.12, but succeeds on 3.8/3.9. TODO investigate.")  # TODO
    def test_output_results(self) -> None:
        """Partitions should be the same."""
        for partitioner_class, expected_partition in self.expected_partition_results.items():
            with self.subTest(partitioner_class=partitioner_class.__name__):
                if issubclass(partitioner_class, partitioners.KMedoidsPartitioner):
                    partitioner = partitioner_class(
                        train_x=self.train_x, kernel=self.kernel, n_experts=self.n_experts, seed=self.seed
                    )
                else:
                    partitioner = partitioner_class(train_x=self.train_x, n_experts=self.n_experts, seed=self.seed)
                observed_partition = partitioner.create_partition()
                self.assertListEqual(expected_partition, observed_partition)

    @unittest.skip("Fails on 3.12, but succeeds on 3.8/3.9. TODO investigate.")  # TODO
    def test_output_results_with_communication(self) -> None:
        """Partitions should be the same."""
        for partitioner_class, expected_partition in self.expected_communication_partition_results.items():
            with self.subTest(partitioner_class=partitioner_class.__name__):
                if issubclass(partitioner_class, partitioners.KMedoidsPartitioner):
                    partitioner = partitioner_class(
                        train_x=self.train_x,
                        kernel=self.kernel,
                        communication=True,
                        n_experts=self.n_experts,
                        seed=self.seed,
                    )
                else:
                    partitioner = partitioner_class(
                        train_x=self.train_x, communication=True, n_experts=self.n_experts, seed=self.seed
                    )
                observed_partition = partitioner.create_partition()
                self.assertListEqual(expected_partition, observed_partition)
