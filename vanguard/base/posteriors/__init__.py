"""
Posterior classes.

Vanguard contains classes to represent posterior distributions, which are used
to encapsulate the predictive posterior of a model at some input points.
"""

from .collection import MonteCarloPosteriorCollection
from .posterior import Posterior

__all__ = ["MonteCarloPosteriorCollection", "Posterior"]
