"""
It is possible to convert a regression problem into a classification problem, allowing the use of Gaussian processes.
"""

from .binary import BinaryClassification
from .categorical import CategoricalClassification
from .dirichlet import DirichletMulticlassClassification
from .mixin import ClassificationMixin

__all__ = [
    "BinaryClassification",
    "CategoricalClassification",
    "DirichletMulticlassClassification",
    "ClassificationMixin",
]
