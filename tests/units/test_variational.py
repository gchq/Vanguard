"""
Test the behaviour of the VariationalInference decorator.
"""
import unittest

from gpytorch.mlls import ExactMarginalLogLikelihood as InappropriateMarginalLogLikelihood

from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference


@VariationalInference()
class VariationalGPController(GaussianGPController):
    pass


class BasicTests(unittest.TestCase):
    """
    Basic tests for the decorator.
    """
    def test_bad_marginal_log_likelihood(self):
        """Should raise a ValueError."""
        dataset = SyntheticDataset()
        with self.assertRaises(ValueError):
            VariationalGPController(dataset.train_x, dataset.train_y, ScaledRBFKernel, dataset.train_y_std,
                                    marginal_log_likelihood_class=InappropriateMarginalLogLikelihood)
