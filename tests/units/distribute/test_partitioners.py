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
Tests for partitioner classes.
"""

import unittest
from unittest.mock import ANY, MagicMock, Mock, patch

import kmedoids
import matplotlib
import numpy as np
import sklearn.cluster
import torch.testing
from gpytorch.kernels import RBFKernel

from tests.cases import assert_mock_called_once_with, get_default_rng
from vanguard.distribute import partitioners


class MockedPartitionTests(unittest.TestCase):
    """
    Test usage of the partition code.

    We verify the steps carried out by the partition methods are correct, but avoid a dependence on
    external packages and random seed across python versions, so do not verify against actual
    clustering or sampling results for any given random seed and dataset.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.train_x = 10.0 * np.arange(10).reshape(-1, 1)
        self.kernel = RBFKernel()
        self.n_experts = 3
        self.rng = get_default_rng()

        # Setup for mocked clustering
        self.example_labels = [0, 0, 1, 0, 1, 1, 2, 1]
        self.expected_partition = [
            # Index of points with label 0 in example_labels
            [0, 1, 3],
            # Index of points with label 1 in example_labels
            [2, 4, 5, 7],
            # Index of points with label 2 in example_labels
            [6],
        ]

    def test_random_sample(self) -> None:
        """
        Test generation of partitions using the random sample method.

        We mock the random choice method used within the code to avoid random-seed dependence.
        """
        # With the RandomPartitioner, we should sample values uniformly at random from
        # [0, 1, ..., num data points-1] without replacement. We have 3 experts and 10 data points.
        # Set a known output to the random choice - this should be a num experts (3) by
        # num data points / num experts (10 / 3 which rounds to 3) array
        mock_rng = Mock(np.random.Generator)
        mock_rng.choice = Mock(return_value=np.array([[8, 1, 5], [0, 7, 2], [9, 4, 3]]))

        partitioner = partitioners.RandomPartitioner(train_x=self.train_x, n_experts=self.n_experts, rng=mock_rng)

        # The partition created should be exactly the random choice result we have specified,
        # no other processing on the data should occur
        self.assertListEqual([[8, 1, 5], [0, 7, 2], [9, 4, 3]], partitioner.create_partition())

    @patch.object(sklearn.cluster, "KMeans")
    def test_k_means(self, mock_clustering) -> None:
        """
        Test generation of partitions using the k-means clustering method.

        We mock the clustering method used within the code to avoid random-seed dependence.
        """
        partitioner = partitioners.KMeansPartitioner(train_x=self.train_x, n_experts=self.n_experts, rng=self.rng)

        # When creating a partition, we should create a clustering object, fit it to the training data,
        # then creation partitions based on those cluster labels
        mocked_clustering_return = MagicMock()
        mocked_fit = MagicMock()
        mocked_fit_return = MagicMock()
        mocked_fit_return.labels_ = np.asarray(self.example_labels)
        mocked_fit.return_value = mocked_fit_return
        mocked_clustering_return.fit = mocked_fit
        mock_clustering.return_value = mocked_clustering_return

        # When we create the partition, we should have one partition per cluster label group,
        # so verify that all data-points with a label of 0 are assigned to a single cluster, and
        # the same for all other labels
        self.assertListEqual(self.expected_partition, partitioner.create_partition())

        # Verify the expected calls were made to the mocked objects - i.e. the clustering was done
        # as expected. We use ANY for the random state as it's derived from the RNG but in a way that's
        # hard to mock out.
        assert_mock_called_once_with(mock_clustering, n_clusters=self.n_experts, random_state=ANY)
        assert_mock_called_once_with(mocked_fit, torch.as_tensor(self.train_x))

    @patch.object(sklearn.cluster, "MiniBatchKMeans")
    def test_mini_batch_k_means(self, mock_clustering) -> None:
        """
        Test generation of partitions using the mini-batch k-means clustering method.

        We mock the clustering method used within the code to avoid random-seed dependence.
        """
        partitioner = partitioners.MiniBatchKMeansPartitioner(
            train_x=self.train_x, n_experts=self.n_experts, rng=self.rng
        )

        # When creating a partition, we should create a clustering object, fit it to the training data,
        # then creation partitions based on those cluster labels
        mocked_clustering_return = MagicMock()
        mocked_fit = MagicMock()
        mocked_fit_return = MagicMock()
        mocked_fit_return.labels_ = self.example_labels
        mocked_fit.return_value = mocked_fit_return
        mocked_clustering_return.fit = mocked_fit
        mock_clustering.return_value = mocked_clustering_return

        # When we create the partition, we should have one partition per cluster label group,
        # so verify that all data-points with a label of 0 are assigned to a single cluster, and
        # the same for all other labels
        self.assertListEqual(self.expected_partition, partitioner.create_partition())

        # Verify the expected calls were made to the mocked objects - i.e. the clustering was done
        # as expected.  We use ANY for the random state as it's derived from the RNG but in a way that's
        # hard to mock out.
        mock_clustering.assert_called_once_with(n_clusters=self.n_experts, random_state=ANY)
        assert_mock_called_once_with(mocked_fit, torch.as_tensor(self.train_x))

    @patch.object(kmedoids, "KMedoids")
    @patch.object(partitioners.KMedoidsPartitioner, "_construct_distance_matrix")
    def test_k_medoids(self, mock_construct_distance_matrix, mock_clustering) -> None:
        """
        Test generation of partitions using the k-medoids clustering method.

        We mock the clustering method used within the code to avoid random-seed dependence.
        """
        # Setup a mocked kernel that returns trivial distances to check
        mocked_kernel = MagicMock()
        mocked_kernel.__class__ = RBFKernel
        actual_distances = np.exp(np.array([0, 0, 1, 1, 0]))
        partitioner = partitioners.KMedoidsPartitioner(
            train_x=self.train_x, n_experts=self.n_experts, kernel=mocked_kernel, rng=self.rng
        )

        # When creating a partition, we should create a clustering object, fit it to the training data,
        # then creation partitions based on those cluster labels
        mocked_clustering_return = MagicMock()
        mocked_fit = MagicMock()
        mocked_fit_return = MagicMock()
        mocked_fit_return.labels_ = np.array(self.example_labels)
        mocked_fit.return_value = mocked_fit_return
        mocked_clustering_return.fit = mocked_fit
        mock_clustering.return_value = mocked_clustering_return
        mock_construct_distance_matrix.return_value = actual_distances

        # When we create the partition, we should have one partition per cluster label group,
        # so verify that all data-points with a label of 0 are assigned to a single cluster, and
        # the same for all other labels
        self.assertListEqual(self.expected_partition, partitioner.create_partition())

        # Verify the expected calls were made to the mocked objects - i.e. the clustering was done
        # as expected. We use ANY for the random state as it's derived from the RNG but in a way that's
        # hard to mock out.
        mock_clustering.assert_called_once_with(n_clusters=self.n_experts, metric="precomputed", random_state=ANY)
        assert_mock_called_once_with(mocked_fit, actual_distances)

    def test_construct_distance_matrix(self):
        """
        Test construction of a distance matrix.

        The distance matrix should be a square matrix that is the result from evaluating the kernel on each
        pair of points.
        """
        partitioner = partitioners.KMedoidsPartitioner(
            train_x=self.train_x, n_experts=self.n_experts, kernel=RBFKernel(), rng=self.rng
        )

        # Compute expected distances in the most clear way possible
        expected_distance_matrix = np.zeros([self.train_x.shape[0], self.train_x.shape[0]])
        for first_index in range(self.train_x.shape[0]):
            for second_index in range(self.train_x.shape[0]):
                expected_distance_matrix[first_index, second_index] = np.exp(
                    -0.5 * (self.train_x[first_index, 0] - self.train_x[second_index, 0]) ** 2
                )
        expected_distance_matrix = np.exp(-expected_distance_matrix)

        # Check output matches expected - note that pylint flags access to a private method, but we
        # want to test it gives sensible results here
        # pylint: disable-next=protected-access
        np.testing.assert_allclose(expected_distance_matrix, partitioner._construct_distance_matrix())

    @patch.object(sklearn.manifold, "TSNE")
    @patch.object(matplotlib.pyplot, "scatter")
    def test_base_partitioner(self, mock_scatter, mock_tsne):
        """
        Test functionality of `BasePartitioner` that is not hit elsewhere in the test suite.

        We verify that trying to create a partition with `BasePartitioner` correctly fails, and plotting
        functionality works as intended.
        """
        partitioner = partitioners.BasePartitioner(train_x=self.train_x, n_experts=self.n_experts, rng=self.rng)

        # If we try to partition with this, we should not be able to as the actual partition methods are implemented in
        # child classes
        with self.assertRaises(NotImplementedError):
            partitioner.create_partition()

        # If we try to plot with this method, we should reduce dimensionality using TSNE and then generate a scatter
        # plot
        mock_tsne_return = MagicMock()
        mock_fit_transform = MagicMock()
        embedding_return = np.array([[0, 1], [2, 3]])
        mock_fit_transform.return_value = embedding_return
        mock_tsne_return.fit_transform = mock_fit_transform

        mock_tsne.return_value = mock_tsne_return

        # Create the plot
        partitioner.plot_partition(partition=[[0, 2], [1, 3]], cmap="Set2")

        # Check mocked calls
        assert_mock_called_once_with(mock_fit_transform, torch.as_tensor(self.train_x))
        self.assertEqual(1, mock_scatter.call_count)
        np.testing.assert_array_equal(mock_scatter.call_args_list[0][0][0], embedding_return[:, 0])
        np.testing.assert_array_equal(mock_scatter.call_args_list[0][0][1], embedding_return[:, 1])

        # Note that we set partition=[[0, 2], [1, 3]] so fill in colour map, extra values are -1
        self.assertDictEqual(
            mock_scatter.call_args_list[0][1], {"c": [0, 1, 0, 1, -1, -1, -1, -1, -1, -1], "cmap": "Set2"}
        )
