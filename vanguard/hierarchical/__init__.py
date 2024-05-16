"""
Enable Bayesian hyperparameters in a Gaussian process.

The :mod:`~vanguard.hierarchical` module contains decorators
to implement Bayesian treatment of hyperparameters using variational inference,
as seen in :cite:`Lalchand20` and [CITATION NEEDED]_ and Laplace approximation
treatment as seen in [CITATION NEEDED]_.
"""
from .laplace import LaplaceHierarchicalHyperparameters
from .module import BayesianHyperparameters
from .variational import VariationalHierarchicalHyperparameters

__all__ = [
    "LaplaceHierarchicalHyperparameters",
    "BayesianHyperparameters",
    "VariationalHierarchicalHyperparameters",
]
