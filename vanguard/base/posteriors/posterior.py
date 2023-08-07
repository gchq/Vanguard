"""
Contains the Posterior class.
"""
import gpytorch
import numpy as np
from scipy import stats
import torch


class Posterior:
    """
    Represents a posterior predictive distribution over a collection of points.

    .. note::
        Various Vanguard decorators are expected to overwrite the :py:meth:`prediction`
        and :py:meth:`confidence_interval` methods of this class. However, the
        :py:meth:`_tensor_prediction` and :py:meth:`_tensor_confidence_interval` methods
        should remain untouched, in order to avoid accidental double transformations.
    """
    def __init__(self, distribution):
        """
        Initialise self.

        :param gpytorch.distributions.MultivariateNormal distribution: The distribution.
        """
        self.distribution = self._add_jitter(distribution)

    @property
    def condensed_distribution(self):
        """
        Return the condensed distribution.

        Return a representative distribution of the posterior, with 1-dimensional
        mean and 2-dimensional covariance.  In standard cases, this will just return
        the distribution.

        :rtype: gpytorch.distributions.MultivariateNormal
        """
        return self.distribution

    def prediction(self):
        """
        Return the prediction as a numpy array.

        :returns: (``means``, ``covar``) where:

            * ``means``: (n_preds,) The posterior predictive mean,
            * ``covar``: (n_preds, n_preds) The posterior predictive covariance matrix.

        :rtype: tuple[numpy.ndarray]
        """
        mean, covar = self._tensor_prediction()
        return mean.detach().cpu().numpy(), covar.detach().cpu().numpy()

    def confidence_interval(self, alpha=0.05):
        """
        Construct confidence intervals around mean of predictive posterior.

        :param float alpha: The significance level of the CIs.
        :returns: The (``median``, ``lower``, ``upper``) bounds of the confidence interval for the
                    predictive posterior, each of shape (n_preds,).
        :rtype: tuple[numpy.ndarray]
        """
        median, lower, upper = self._tensor_confidence_interval(alpha)
        return median.detach().cpu().numpy(), lower.detach().cpu().numpy(), upper.detach().cpu().numpy()

    def mse(self, y):
        r"""
        Compute the mean-squared of some values under the posterior.

        :param numpy.ndarray y: (n, d) or (d,) where d is the dimension of the space on which the
            posterior is defined. Sum over first dimension if two dimensional.
        :returns: The MSE of the given y values, i.e. :math:`\frac{1}{n}\sum_{i} (y_i - \hat{y}_i)`.
        :rtype: float
        """
        mean, _ = self.prediction()
        return ((mean - y)**2).mean()

    def nll(self, y, noise_variance=0, alpha=stats.norm.cdf(-1)*2):
        """
        Compute the negative log-likelihood of some values under the posterior.

        :param numpy.ndarray y: (n, d) or (d,) where d is the dimension of the space on which the
            posterior is defined. Sum over first dimension if two dimensional.
        :param numpy.ndarray,float noise_variance: Additional variance to be included in the calculation.
        :param float alpha: The significance of the confidence interval used to calculate the standard deviation.
        :returns: The negative log-likelihood of the given y values.
        :rtype: float
        """
        mean, _, upper = self.confidence_interval(alpha)
        variance = (upper - mean) ** 2
        sigma = variance + noise_variance
        rss = (y - mean) ** 2
        const = 0.5 * np.log(2 * np.pi * sigma)
        p_nll = const + rss / (2 * sigma)
        return p_nll.mean()

    def log_probability(self, y):
        r"""
        Compute the log-likelihood of some values under the posterior.

        :param numpy.ndarray y: (n, d) or (d,) where d is the dimension of the space on which the
            posterior is defined. Sum over first dimension if two dimensional.
        :returns: The log-likelihood of the given y values, i.e. :math:`\sum_{i} \log P(y_i)`
            where :math:`P` is the posterior density.
        :rtype: float
        """
        return self._tensor_log_probability(torch.as_tensor(y).float()).item()

    def sample(self, n_samples=1):
        """Draw independent samples from the posterior."""
        return self._tensor_sample(sample_shape=torch.Size([n_samples])).detach().cpu().numpy()

    @classmethod
    def from_mean_and_covariance(cls, mean, covariance):
        """
        Construct from the mean and covariance of a Gaussian.

        :param torch.Tensor mean: (d,) or (d, t) The mean of the Gaussian.
        :param torch.Tensor covariance: (d, d) or (dt, dt) The covariance matrix of the Gaussian.
        :returns: The multivariate Gaussian distribution for either a single task or multiple tasks, depending on the
                  shape of the args.
        :rtype: gpytorch.distributions.MultivariateNormal
        """
        return cls(cls._make_multivariate_normal(mean, covariance))

    def _tensor_prediction(self):
        """
        Return the prediction as a tensor.

        .. warning::
            Overwriting this method is not safe, as it may affect the transformations applied by
            certain decorators. Consider overwriting :py:meth:`prediction` instead.

        :returns: (``means``, ``covar``) where:

            * ``means``: (n_preds,) The posterior predictive mean,
            * ``covar``: (n_preds, n_preds) The posterior predictive covariance matrix.

        :rtype: tuple[torch.tensor]
        """
        try:
            covar = self.distribution.covariance_matrix
        except AttributeError:
            covar = torch.diag(self.distribution.variance)
        return self.distribution.mean, covar

    def _tensor_confidence_interval(self, alpha):
        """
        Construct confidence intervals around mean of predictive posterior.

        .. warning::
            Overwriting this method is not safe, as it may affect the transformations applied by
            certain decorators. Consider overwriting :py:meth:`confidence_interval` instead.

        :param float alpha: The significance level of the CIs.
        :returns: The (``median``, ``lower``, ``upper``) bounds of the confidence interval for the
                    predictive posterior, each of shape (n_preds,).
        :rtype: tuple[torch.tensor]
        """
        mean, covar = self._tensor_prediction()
        return self._gaussian_confidence_interval(mean, covar, alpha=alpha)

    def _tensor_sample(self, sample_shape=torch.Size()):
        """Return samples as a tensor."""
        return self.distribution.rsample(sample_shape=sample_shape)

    def _tensor_sample_condensed(self, sample_shape=torch.Size()):
        """Return samples from the condensed distribution as a tensor."""
        return self.condensed_distribution.rsample(sample_shape=sample_shape)

    def _tensor_log_probability(self, y):
        r"""
        Compute the log-likelihood of some values under the posterior.

        .. warning::
            Overwriting this method is not safe, as it may affect the transformations applied by
            certain decorators. Consider overwriting :py:meth:`log_probability` instead.

        :param torch.Tensor y: (n, d) or (d,) where d is the dimension of the space on which the
            posterior is defined. Sum over first dimension if two dimensional.
        :returns: The log-likelihood of the given y values, i.e. :math:`\sum_{i} \log P(y_i)`
            where :math:`P` is the posterior density.
        :rtype: torch.Tensor
        """
        return self.distribution.log_prob(y.contiguous()).sum()

    @staticmethod
    def _make_multivariate_normal(mean, covariance):
        mean = mean.squeeze(dim=-1)
        if mean.ndim == 1:
            distribution = gpytorch.distributions.MultivariateNormal(mean, covariance)
        else:
            distribution = gpytorch.distributions.MultitaskMultivariateNormal(mean, covariance)
        return distribution

    @staticmethod
    def _gaussian_confidence_interval(mean, covariance, alpha=0.05):
        """
        Get pointwise (diagonal) confidence intervals for a multivariate Gaussian's coordinates.

        If the Gaussian is "multi-task", then a confidence interval is computed for each task.

        :param torch.Tensor mean: (d,) or (d, t) The mean of the Gaussian.
        :param torch.Tensor covariance: (d,d) or (d*t, d*t) The covariance matrix of the Gaussian.
        :param float alpha: The significance of the interval.
        :return: The (``median``, ``lower``, ``upper``) bounds of the confidence interval each of shape (d,) or (d,t)
        :rtype: tuple[torch.Tensor]
        """
        stds = torch.sqrt(torch.diag(covariance))
        try:
            num_tasks = mean.shape[1]
        except IndexError:
            num_tasks = 1
            mean = mean.unsqueeze(dim=-1)
        num_points = mean.shape[0]
        stds = torch.stack([stds[num_points*i: num_points*(i+1)] for i in range(num_tasks)], -1)
        conf_factor = stats.norm.ppf(1 - alpha / 2)
        median = mean
        lower = mean - stds * conf_factor
        upper = mean + stds * conf_factor

        return median.squeeze(dim=-1), lower.squeeze(dim=-1), upper.squeeze(dim=-1)

    @staticmethod
    def _add_jitter(distribution):
        """
        Add diagonal jitter to covariance matrices to avoid indefinite covariance matrices.

        :param gpytorch.distributions.MultivariateNormal distribution: The distribution to be jittered.
        :returns: The given distribution with a new covariance matrix including some jitter.
        :rtype: gpytorch.distributions.MultivariateNormal
        """
        try:
            covar = distribution.covariance_matrix
        except AttributeError:
            return distribution
        jitter = gpytorch.settings.cholesky_jitter.value(covar.dtype) * 10
        return distribution.add_jitter(jitter)
