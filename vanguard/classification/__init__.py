"""
It is possible to convert a regression problem into a classification problem, allowing the use of Gaussian processes.
"""

from vanguard.classification.binary import BinaryClassification
from vanguard.classification.categorical import CategoricalClassification
from vanguard.classification.dirichlet import DirichletMulticlassClassification

__all__ = [
    "BinaryClassification",
    "CategoricalClassification",
    "DirichletMulticlassClassification",
]
