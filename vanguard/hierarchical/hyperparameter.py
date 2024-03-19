"""
Contains the BayesianHyperparameter class.
"""
from gpytorch import constraints
import torch
from typing import Union


class BayesianHyperparameter:
    """
    Represents a single Bayesian hyperparameter.
    """
    def __init__(self, raw_name: str, raw_shape: torch.Size, constraint: Union[constraints.Interval, None], prior_mean: float, prior_variance: float):
        """
        Initialise self.

        :param raw_name: The raw name for the parameter.
        :param raw_shape: The shape of the raw parameter.
        :param constraint: The constraint (if any) placed on the parameter.
        :param prior_mean: The mean of the diagonal normal prior on the raw parameter.
        :param prior_variance: The variance of the diagonal normal prior on the raw parameter.
        """
        self.raw_name = raw_name
        self.raw_shape = raw_shape
        self.constraint = constraint
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

    def numel(self) -> int:
        """Return the number of elements in the parameter."""
        return self.raw_shape.numel()
