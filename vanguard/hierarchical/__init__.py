"""
Enable Bayesian hyperparameters in a Gaussian process.

The :mod:`~vanguard.hierarchical` module contains decorators
to implement Bayesian treatment of hyperparameters using variational inference,
as seen in :cite:`Lalchand20`, as well as Laplace approximation
treatment.
"""

from .laplace import LaplaceHierarchicalHyperparameters
from .module import BayesianHyperparameters
from .variational import VariationalHierarchicalHyperparameters

__all__ = [
    "LaplaceHierarchicalHyperparameters",
    "BayesianHyperparameters",
    "VariationalHierarchicalHyperparameters",
]
