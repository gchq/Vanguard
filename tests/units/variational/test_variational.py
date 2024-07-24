"""
Test the behaviour of the VariationalInference decorator.
"""

import unittest

from gpytorch.mlls import ExactMarginalLogLikelihood as InappropriateMarginalLogLikelihood

from tests.cases import get_default_rng
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

    def test_bad_marginal_log_likelihood(self) -> None:
        """
        Ensure that the underlying TypeError is converted to a ValueError.
        """
        rng = get_default_rng()
        dataset = SyntheticDataset(rng=rng)
        with self.assertRaises(ValueError):
            VariationalGPController(
                dataset.train_x,
                dataset.train_y,
                ScaledRBFKernel,
                dataset.train_y_std,
                marginal_log_likelihood_class=InappropriateMarginalLogLikelihood,
                rng=rng,
            )
