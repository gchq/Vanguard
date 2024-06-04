"""
Basic end to end functionality test for distributed decorators in Vanguard.
"""

import unittest

import numpy as np
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from sklearn.metrics import f1_score

from vanguard.classification import BinaryClassification
from vanguard.distribute import Distributed
from vanguard.distribute.aggregators import (
    BCMAggregator,
    EKPOEAggregator,
    GPOEAggregator,
    GRBCMAggregator,
    RBCMAggregator,
    XBCMAggregator,
    XGRBCMAggregator,
)
from vanguard.distribute.partitioners import (
    KMeansPartitioner,
    MiniBatchKMeansPartitioner,
    RandomPartitioner,
)
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference


class VanguardTestCase(unittest.TestCase):
    """
    A subclass of TestCase designed to check end-to-end usage of distributed code.
    """

    def setUp(self) -> None:
        """
        Define data shared across tests.
        """
        self.random_seed = 1_989
        self.num_train_points = 100
        self.num_test_points = 100
        self.n_sgd_iters = 50
        # How high of an F1 score do we need to consider the test a success (and a fit
        # successful?)
        self.required_f1_score = 0.9

    def test_distributed_gp_vary_aggregator_fix_partition(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable distributed binary classification problem
        using the various aggregators but a fixed partition method.

        We generate a single feature `x` and a binary target `y`, and verify that a
        GP can be fit to this data.
        """
        np.random.seed(self.random_seed)

        # Define some data for the test
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.zeros_like(x)
        for index, x_val in enumerate(x):
            # Set some non-trivial classification target
            if 0.25 < x_val < 0.5:
                y[index, 0] = 1
            if x_val > 0.8:
                y[index, 0] = 1

        # Split data into training and testing
        train_indices = np.random.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # We have a binary classification problem, so we apply the BinaryClassification
        # decorator and will need to use VariationalInference to perform inference on
        # data. We try each aggregation method to ensure they are all functional.
        for aggregator in [
            EKPOEAggregator,
            GPOEAggregator,
            BCMAggregator,
            RBCMAggregator,
            XBCMAggregator,
            GRBCMAggregator,
            XGRBCMAggregator,
        ]:

            @Distributed(n_experts=3, aggregator_class=aggregator)
            @BinaryClassification()
            @VariationalInference()
            class BinaryClassifier(GaussianGPController):
                pass

            # Define the controller object
            gp = BinaryClassifier(
                train_x=x[train_indices],
                train_y=y[train_indices],
                kernel_class=ScaledRBFKernel,
                y_std=0,
                likelihood_class=BernoulliLikelihood,
                marginal_log_likelihood_class=VariationalELBO,
            )

            # Fit the GP
            gp.fit(n_sgd_iters=self.n_sgd_iters)

            # Get predictions from the controller object
            predictions_train, _ = gp.classify_points(x[train_indices])
            predictions_test, _ = gp.classify_points(x[test_indices])

            # Sense check outputs
            self.assertGreaterEqual(f1_score(predictions_train, y[train_indices]), self.required_f1_score)
            self.assertGreaterEqual(f1_score(predictions_test, y[test_indices]), self.required_f1_score)

    def test_distributed_gp_vary_partition_fix_aggregator(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable distributed binary classification problem
        using the various partition methods but a fixed aggregation method.

        We generate a single feature `x` and a binary target `y`, and verify that a
        GP can be fit to this data.
        """
        np.random.seed(self.random_seed)

        # Define some data for the test
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.zeros_like(x)
        for index, x_val in enumerate(x):
            # Set some non-trivial classification target
            if 0.25 < x_val < 0.5:
                y[index, 0] = 1
            if x_val > 0.8:
                y[index, 0] = 1

        # Split data into training and testing
        train_indices = np.random.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # We have a binary classification problem, so we apply the BinaryClassification
        # decorator and will need to use VariationalInference to perform inference on
        # data. We try each partition method to ensure they are all functional.
        for partitioner in [
            RandomPartitioner,
            KMeansPartitioner,
            MiniBatchKMeansPartitioner,
            # TODO: Do we have an example of this working?
            # KMedoidsPartitioner,
        ]:

            @Distributed(n_experts=3, partitioner_class=partitioner)
            @BinaryClassification()
            @VariationalInference()
            class BinaryClassifier(GaussianGPController):
                pass

            # Define the controller object
            gp = BinaryClassifier(
                train_x=x[train_indices],
                train_y=y[train_indices],
                kernel_class=ScaledRBFKernel,
                y_std=0,
                likelihood_class=BernoulliLikelihood,
                marginal_log_likelihood_class=VariationalELBO,
            )

            # Fit the GP
            gp.fit(n_sgd_iters=self.n_sgd_iters)

            # Get predictions from the controller object
            predictions_train, _ = gp.classify_points(x[train_indices])
            predictions_test, _ = gp.classify_points(x[test_indices])

            # Sense check outputs
            self.assertGreaterEqual(f1_score(predictions_train, y[train_indices]), self.required_f1_score)
            self.assertGreaterEqual(f1_score(predictions_test, y[test_indices]), self.required_f1_score)


if __name__ == "__main__":
    unittest.main()
