"""
Enable Bayesian hyperparameters in a Gaussian process.

The :mod:`~vanguard.hierarchical` module contains decorators
to implement Bayesian treatment of hyperparameters using variational inference,
as seen in :cite:`Lalchand20`, as well as Laplace approximation
treatment.
"""

from vanguard.hierarchical.laplace import LaplaceHierarchicalHyperparameters
from vanguard.hierarchical.module import BayesianHyperparameters
from vanguard.hierarchical.variational import VariationalHierarchicalHyperparameters

__all__ = [
    "LaplaceHierarchicalHyperparameters",
    "BayesianHyperparameters",
    "VariationalHierarchicalHyperparameters",
]
