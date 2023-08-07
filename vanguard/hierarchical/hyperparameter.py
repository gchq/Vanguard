"""
Contains the BayesianHyperparameter class.
"""


class BayesianHyperparameter:
    """
    Represents a single Bayesian hyperparameter.
    """
    def __init__(self, raw_name, raw_shape, constraint, prior_mean, prior_variance):
        """
        Initialise self.

        :param str raw_name: The raw name for the parameter.
        :param torch.Size raw_shape: The shape of the raw parameter.
        :param gpytorch.constraints.Interval,None, constraint: The constraint (if any) placed on the parameter.
        :param float prior_mean: The mean of the diagonal normal prior on the raw parameter.
        :param float prior_variance: The variance of the diagonal normal prior on the raw parameter.
        """
        self.raw_name = raw_name
        self.raw_shape = raw_shape
        self.constraint = constraint
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

    def numel(self):
        """Return the number of elements in the parameter."""
        return self.raw_shape.numel()
