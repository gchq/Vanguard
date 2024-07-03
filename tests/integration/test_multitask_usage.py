"""
Basic end to end functionality test for multitask problems in Vanguard.
"""

import unittest

import numpy as np
from gpytorch.mlls import VariationalELBO
from sklearn.metrics import f1_score

from vanguard.classification import BinaryClassification
from vanguard.classification.likelihoods import MultitaskBernoulliLikelihood
from vanguard.kernels import ScaledRBFKernel
from vanguard.multitask import Multitask
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference


class VanguardTestCase(unittest.TestCase):
    """
    A subclass of TestCase designed to check end-to-end usage of classification code.
    """

    def setUp(self) -> None:
        """
        Define data shared across tests.
        """
        self.rng = np.random.default_rng(1_989)
        self.num_train_points = 100
        self.num_test_points = 100
        self.n_sgd_iters = 500
        # How high of an F1 score do we need to consider the test a success (and a fit
        # successful?)
        self.required_f1_score = 0.5

    @unittest.skip  # TODO: Fix test - unacceptably flaky
    # https://github.com/gchq/Vanguard/issues/141
    def test_gp_multitask_binary_classification(self) -> None:
        """
        Verify Vanguard usage on a multitask binary classification problem.

        We generate a single feature `x` and a multivariate target `y`, and verify that a
        GP can be fit to this data.
        """
        # Define some data for the test
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.zeros([x.shape[0], 2])
        for index, x_val in enumerate(x):
            # Set some non-trivial classification targets
            if 0.25 < x_val < 0.5:
                y[index, 0] = 1
            if x_val < 0.3:
                y[index, 1] = 1

        # Split data into training and testing
        train_indices = self.rng.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # We have a multitask binary classification problem, so we apply the BinaryClassification
        # decorator, Multitask with 2 tasks (the number of columns in y) and will need to use
        # VariationalInference to perform inference on data
        @BinaryClassification()
        @Multitask(num_tasks=2)
        @VariationalInference()
        class MultiTaskCategoricalClassifier(GaussianGPController):
            pass

        # Define the controller object
        gp = MultiTaskCategoricalClassifier(
            train_x=x[train_indices],
            train_y=y[train_indices],
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=MultitaskBernoulliLikelihood,
            marginal_log_likelihood_class=VariationalELBO,
        )

        # Fit the GP
        gp.fit(n_sgd_iters=self.n_sgd_iters)

        # Get predictions from the controller object
        predictions_train, _ = gp.classify_points(x[train_indices])
        predictions_test, _ = gp.classify_points(x[test_indices])

        # Verify performance
        for task_index in range(y.shape[1]):
            self.assertGreaterEqual(
                f1_score(predictions_train[:, task_index], y[train_indices, task_index]), self.required_f1_score
            )
            self.assertGreaterEqual(
                f1_score(predictions_test[:, task_index], y[test_indices, task_index]), self.required_f1_score
            )


if __name__ == "__main__":
    unittest.main()
