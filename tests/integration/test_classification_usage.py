"""
Basic end to end functionality test for classification problems in Vanguard.
"""

import unittest

import numpy as np
from gpytorch.likelihoods import BernoulliLikelihood, DirichletClassificationLikelihood
from gpytorch.mlls import VariationalELBO
from sklearn.metrics import f1_score

from vanguard.classification import BinaryClassification, DirichletMulticlassClassification
from vanguard.kernels import ScaledRBFKernel
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
        self.random_seed = 1_989
        self.num_train_points = 100
        self.num_test_points = 100
        self.n_sgd_iters = 50
        # How high of an F1 score do we need to consider the test a success (and a fit
        # successful?)
        self.required_f1_score = 0.95

    def test_gp_binary_classification(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable binary classification problem.

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
        # data
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

    def test_gp_categorical_classification_exact(self) -> None:
        """
        Verify Vanguard usage on a simple, single variable categorical classification problem
        using exact inference.

        We generate a single feature `x` and a categorical target `y`, and verify that a
        GP can be fit to this data.
        """
        np.random.seed(self.random_seed)

        # Define some data for the test
        x = np.linspace(start=0, stop=10, num=self.num_train_points + self.num_test_points).reshape(-1, 1)
        y = np.zeros_like(x)
        for index, x_val in enumerate(x):
            # Set some non-trivial classification target with 3 classes (0, 1 and 2)
            if 0.25 < x_val < 0.5:
                y[index, 0] = 1
            if x_val > 0.8:
                y[index, 0] = 2

        # Split data into training and testing
        train_indices = np.random.choice(np.arange(y.shape[0]), size=self.num_train_points, replace=False)
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)

        # We have a multi-class classification problem, so we apply the DirichletMulticlassClassification
        # decorator to perform exact inference
        @DirichletMulticlassClassification(num_classes=3)
        class CategoricalClassifier(GaussianGPController):
            pass

        # TODO: This currently fails, is it a python version issue or a Vanguard issue?
        #   Note that if we use the example directly from the decorator definition:
        #   train_x=np.array([0, 0.1, 0.45, 0.55, 0.9, 1]),
        #   train_y=np.array([0, 0, 1, 1, 2, 2]),
        #   We get the same error, so this might be an issue with the underlying code
        # Define the controller object
        gp = CategoricalClassifier(
            train_x=x[train_indices],
            train_y=y[train_indices, 0],
            kernel_class=ScaledRBFKernel,
            y_std=0,
            likelihood_class=DirichletClassificationLikelihood,
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
