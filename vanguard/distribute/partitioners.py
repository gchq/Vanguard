"""
Partitioners are responsible for separating the training data into subsets to be assigned to each expert controller.
"""

from collections import defaultdict
from typing import Iterable, List, Optional, Union

import gpytorch.kernels
import kmedoids
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.manifold
import torch
from matplotlib.colors import Colormap
from numpy.typing import NDArray


# TODO: should this be an abstract base class?
# https://github.com/gchq/Vanguard/issues/198
# TODO: Should we make BasePartitioner generic in the NDArray dtype?
# https://github.com/gchq/Vanguard/issues/198
class BasePartitioner:
    """
    Generate a partition over index space using various methods. All partitioners should inherit from this class.

    :param train_x: The mean of the inputs.
    :param n_experts: The number of partitions in which to split the data. Defaults to 3.
    :param communication: If True, A communications expert will be included. Defaults to False.
    :param seed: The seed for the random state. Defaults to 42.
    """

    def __init__(
        self, train_x: NDArray[np.floating], n_experts: int = 3, communication: bool = False, seed: Optional[int] = 42
    ):
        """
        Initialise the BasePartitioner class.
        """
        self.train_x = train_x
        self.n_experts = n_experts
        self.communication = communication
        self.seed = seed

        self.n_examples = self.train_x.shape[0]

    def create_partition(self) -> List[List[int]]:
        """
        Create a partition of ``self.train_x`` across ``self.n_experts``.

        :return: A partition of length ``self.n_experts``.
        """
        np.random.seed(self.seed)

        if self.communication:
            partition = self._create_cluster_communication_partition()
        else:
            partition = self._create_cluster_partition(self.n_experts)

        return partition

    def plot_partition(
        self, partition: List[List[int]], cmap: Optional[Union[str, Colormap]] = "Set3", **plot_kwargs
    ) -> None:
        """
        Plot a partition on a T-SNE graph.

        :param partition: List of data partitions to plot
        :param cmap: Colormap to use for plotting
        """
        embedding = sklearn.manifold.TSNE().fit_transform(self.train_x)

        colours = [-1 for _ in range(len(self.train_x))]
        for group_index, group in enumerate(partition):
            for data_point_index in group:
                colours[data_point_index] = group_index

        plt.scatter(embedding[:, 0], embedding[:, 1], c=colours, cmap=cmap, **plot_kwargs)

    def _create_cluster_partition(self, n_clusters: int) -> List[List[int]]:
        """
        Create the partition.

        :param n_clusters: The number of clusters.
        :return: A partition of shape (``n_clusters``, ``self.n_examples`` // ``n_clusters``).
        """
        # TODO: should this be an abstract method?
        # https://github.com/gchq/Vanguard/issues/198
        raise NotImplementedError

    def _create_cluster_communication_partition(self) -> List[List[int]]:
        """
        Create a partition with a communications expert.

        :return: A partition of length ``self.n_experts``.
        """
        size = self.n_examples // self.n_experts
        random_partition = np.random.choice(self.n_examples, size=size, replace=False).tolist()
        cluster_partition = self._create_cluster_partition(self.n_experts - 1)

        for i in range(self.n_experts - 1):
            cluster_partition[i] = random_partition + cluster_partition[i]

        partition = [random_partition, *cluster_partition]

        return partition

    @staticmethod
    def _group_indices_by_label(labels: Iterable[int]) -> List[List[int]]:
        """
        Group the indices of the labels by their value.

        :param labels: An array of labels.
        :return: A list of values such that labels[groups[i][j]] == i for all j in groups[i].

        :Example:
            >>> labels = [1, 2, 3, 2, 1, 3, 0, 9]
            >>> BasePartitioner._group_indices_by_label(labels)
            [[6], [0, 4], [1, 3], [2, 5], [], [], [], [], [], [7]]
        """
        label_value_to_index = defaultdict(list)
        for label_index, label_value in enumerate(labels):
            label_value_to_index[label_value].append(label_index)

        groups = [label_value_to_index[value] for value in range(max(labels) + 1)]
        return groups


class RandomPartitioner(BasePartitioner):
    """
    Create a partition using random sampling.
    """

    def _create_cluster_partition(self, n_clusters: int) -> List[List[int]]:
        """
        Create the partition via uniform random sampling.

        :param n_clusters: The number of clusters.
        :return: A partition of shape (``n_clusters``, ``self.n_examples`` // ``n_clusters``).
        """
        size = (n_clusters, self.n_examples // n_clusters)
        partition = np.random.choice(self.n_examples, size=size, replace=False).tolist()
        return partition


class KMeansPartitioner(BasePartitioner):
    """
    Create a partition using K-Means.
    """

    def _create_cluster_partition(self, n_clusters: int) -> List[List[int]]:
        """
        Create the partition by clustering the data using KMeans clustering.

        :param n_clusters: The number of clusters.
        :return: A partition of shape (``n_clusters``, ``self.n_examples`` // ``n_clusters``).
        """
        clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=self.seed)
        labels = clusterer.fit(self.train_x).labels_
        partition = self._group_indices_by_label(labels)
        return partition


class MiniBatchKMeansPartitioner(BasePartitioner):
    """
    Create a partition using Mini-batch K-Means.
    """

    def _create_cluster_partition(self, n_clusters: int) -> List[List[int]]:
        """
        Create the partition by clustering the data using KMeans clustering, but processing data in batches.

        :param n_clusters: The number of clusters.
        :return: A partition of shape (``n_clusters``, ``self.n_examples`` // ``n_clusters``).
        """
        clusterer = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters, random_state=self.seed)
        labels = clusterer.fit(self.train_x).labels_
        partition = self._group_indices_by_label(labels)
        return partition


class KMedoidsPartitioner(BasePartitioner):
    """
    Create a partition using KMedoids with similarity defined by the kernel.

    :param train_x: The mean of the inputs.
    :param kernel: The kernel to use for constructing the
            similarity matrix in KMedoids.
    :param n_experts: The number of partitions in which to split the data. Defaults to 2.
    :param communication: If True, A communications expert will be included. Defaults to False.
    :param seed: The seed for the random state. Defaults to 42.

    :seealso: Clusters are computed using a :class:`kmedoids.KMedoids` object.
    """

    def __init__(
        self,
        train_x: NDArray[np.floating],
        kernel: gpytorch.kernels.Kernel,
        n_experts: int = 2,
        communication: bool = False,
        seed: Optional[int] = 42,
    ):
        """
        Initialise the KMedoidsPartitioner class.
        """
        super().__init__(train_x=train_x, n_experts=n_experts, communication=communication, seed=seed)
        self.kernel = kernel

    def _create_cluster_partition(self, n_clusters: int) -> List[List[int]]:
        """
        Create the partition by clustering the data using KMedoids clustering.

        :param n_clusters: The number of clusters.
        :return: A partition of shape (``n_clusters``, ``self.n_examples`` // ``n_clusters``).
        """
        dist_matrix = self._construct_distance_matrix()
        clusterer = kmedoids.KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=self.seed)
        labels = clusterer.fit(dist_matrix).labels_
        partition = self._group_indices_by_label(labels)
        return partition

    def _construct_distance_matrix(self) -> NDArray[np.floating]:
        """
        Construct the distance matrix, where distance is judged by the kernel set at creation time.

        :return dist_matrix: The distance matrix.

        .. warning::
            The affinity matrix takes up O(N^2) memory so can't be used for large ``train_x``.
        """
        affinity_matrix = self.kernel(torch.from_numpy(self.train_x)).cpu().evaluate().detach().cpu().numpy()
        dist_matrix = np.exp(-affinity_matrix)
        return dist_matrix
