"""
Contains the Posterior class.
"""
import gpytorch
import numpy as np
import numpy.typing
from scipy import stats
import torch
from typing import TypeVar, Tuple, Union
from typing_extensions import Self

T = TypeVar("T")


class Posterior:
    """
    Represents a posterior predictive distribution over a collection of points.

    .. note::
        Various Vanguard decorators are expected to overwrite the :meth:`prediction`
        and :meth:`confidence_interval` methods of this class. However, the
        :meth:`_tensor_prediction` and :meth:`_tensor_confidence_interval` methods
        should remain untouched, in order to avoid accidental double transformations.


    :param distribution: The distribution.
    """
    def __init__(
            self,
            distribution: gpytorch.distributions.Distribution,
    ):
        """Initialise self."""
        self.distribution = self._add_jitter(distribution)

    @property
    def condensed_distribution(self) -> gpytorch.distributions.MultivariateNormal:
        """
        Return the condensed distribution.

        Return a representative distribution of the posterior, with 1-dimensional
        mean and 2-dimensional covariance.  In standard cases, this will just return
        the distribution.
        """
        return self.distribution

    def prediction(self) -> Tuple[numpy.typing.NDArray[float], numpy.typing.NDArray[float]]:
        """
        Return the prediction as a numpy array.

        :returns: (``means``, ``covar``) where:

            * ``means``: (n_preds,) The posterior predictive mean,
            * ``covar``: (n_preds, n_preds) The posterior predictive covariance matrix.
        """
        mean, covar = self._tensor_prediction()
        return mean.detach().cpu().numpy(), covar.detach().cpu().numpy()

    def confidence_interval(
            self,
            alpha: float = 0.05,
    ) -> Tuple[numpy.typing.NDArray[float], numpy.typing.NDArray[float], numpy.typing.NDArrayndarray[float]]:
        """
        Construct confidence intervals around mean of predictive posterior.

        :param alpha: The significance level of the CIs.
        :returns: The (``median``, ``lower``, ``upper``) bounds of the confidence interval for the
                    predictive posterior, each of shape (n_preds,).
        """
        median, lower, upper = self._tensor_confidence_interval(alpha)
        return median.detach().cpu().numpy(), lower.detach().cpu().numpy(), upper.detach().cpu().numpy()

    def mse(
            self,
            y: Union[numpy.typing.NDArray[float], float],
    ) -> float:
        r"""
        Compute the mean-squared of some values under the posterior.

        :param y: (n, d) or (d,) where d is the dimension of the space on which the
            posterior is defined. Sum over first dimension if two dimensional.
        :returns: The MSE of the given y values, i.e. :math:`\frac{1}{n}\sum_{i} (y_i - \hat{y}_i)`.
        """
        mean, _ = self.prediction()
        return ((mean - y)**2).mean()

    def nll(
            self,
            y: Union[numpy.typing.NDArray[float], float],
            noise_variance: Union[numpy.typing.NDArray[float], float] = 0,
            alpha: float = stats.norm.cdf(-1)*2,
    ) -> float:
        """
        Compute the negative log-likelihood of some values under the posterior.

        :param y: (n, d) or (d,) where d is the dimension of the space on which the
            posterior is defined. Sum over first dimension if two dimensional.
        :param noise_variance: Additional variance to be included in the calculation.
        :param alpha: The significance of the confidence interval used to calculate the standard deviation.
        :returns: The negative log-likelihood of the given y values.
        """
        mean, _, upper = self.confidence_interval(alpha)
        variance = (upper - mean) ** 2
        sigma = variance + noise_variance
        rss = (y - mean) ** 2
        const = 0.5 * np.log(2 * np.pi * sigma)
        p_nll = const + rss / (2 * sigma)
        return p_nll.mean()

    def log_probability(
            self,
            y: Union[numpy.typing.NDArray[float], float],
    ) -> float:
        r"""
        Compute the log-likelihood of some values under the posterior.

        :param y: (n, d) or (d,) where d is the dimension of the space on which the
            posterior is defined. Sum over first dimension if two dimensional.
        :returns: The log-likelihood of the given y values, i.e. :math:`\sum_{i} \log P(y_i)`
            where :math:`P` is the posterior density.
        """
        return self._tensor_log_probability(torch.as_tensor(y).float()).item()

    def sample(
            self,
            n_samples: int = 1
    ) -> numpy.typing.NDArray[float]:
        """
        Draw independent samples from the posterior.

        :param n_samples: The number of samples to draw.
        """
        return self._tensor_sample(sample_shape=torch.Size([n_samples])).detach().cpu().numpy()

    @classmethod
    def from_mean_and_covariance(
            cls,
            mean: torch.Tensor,
            covariance: torch.Tensor,
    ) -> Self:
        """
        Construct from the mean and covariance of a Gaussian.

        :param mean: (d,) or (d, t) The mean of the Gaussian.
        :param covariance: (d, d) or (dt, dt) The covariance matrix of the Gaussian.
        :returns: The multivariate Gaussian distribution for either a single task or multiple tasks, depending on the
                  shape of the args.
        """
        return cls(cls._make_multivariate_normal(mean, covariance))

    def _tensor_prediction(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the prediction as a tensor.

        .. warning::
            Overwriting this method is not safe, as it may affect the transformations applied by
            certain decorators. Consider overwriting :meth:`prediction` instead.

        :returns: (``means``, ``covar``) where:

            * ``means``: (n_preds,) The posterior predictive mean,
            * ``covar``: (n_preds, n_preds) The posterior predictive covariance matrix.
        """
        try:
            covar = self.distribution.covariance_matrix
        except AttributeError:
            covar = torch.diag(self.distribution.variance)
        return self.distribution.mean, covar

    def _tensor_confidence_interval(
            self,
            alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct confidence intervals around mean of predictive posterior.

        .. warning::
            Overwriting this method is not safe, as it may affect the transformations applied by
            certain decorators. Consider overwriting :meth:`confidence_interval` instead.

        :param alpha: The significance level of the CIs.
        :returns: The (``median``, ``lower``, ``upper``) bounds of the confidence interval for the
                    predictive posterior, each of shape (n_preds,).
        """
        mean, covar = self._tensor_prediction()
        return self._gaussian_confidence_interval(mean, covar, alpha=alpha)

    def _tensor_sample(
            self,
            sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """
        Return samples as a tensor.

        :param sample_shape: The shape of the sample.
        """
        return self.distribution.rsample(sample_shape=sample_shape)

    def _tensor_sample_condensed(
            self,
            sample_shape=torch.Size()
    ) -> torch.Tensor:
        """
        Return samples from the condensed distribution as a tensor.

        :param sample_shape: The shape of the sample.
        """
        return self.condensed_distribution.rsample(sample_shape=sample_shape)

    def _tensor_log_probability(
            self,
            y: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the log-likelihood of some values under the posterior.

        .. warning::
            Overwriting this method is not safe, as it may affect the transformations applied by
            certain decorators. Consider overwriting :meth:`log_probability` instead.

        :param y: (n, d) or (d,) where d is the dimension of the space on which the
            posterior is defined. Sum over first dimension if two dimensional.
        :returns: The log-likelihood of the given y values, i.e. :math:`\sum_{i} \log P(y_i)`
            where :math:`P` is the posterior density.
        """
        return self.distribution.log_prob(y.contiguous()).sum()

    @staticmethod
    def _make_multivariate_normal(
            mean: torch.Tensor,
            covariance: torch.Tensor,
    ) -> torch.distributions.MultivariateNormal:
        r"""
        Construct MultivariateNormal or MultitaskMultivariateNormal from mean and covariance.

        :param mean: (d,) or (d, t) The mean of the Gaussian.
        :param covariance: (d, d) or (dt, dt) The covariance matrix of the Gaussian.
        """
        mean = mean.squeeze(dim=-1)
        if mean.ndim == 1:
            distribution = gpytorch.distributions.MultivariateNormal(mean, covariance)
        else:
            distribution = gpytorch.distributions.MultitaskMultivariateNormal(mean, covariance)
        return distribution

    @staticmethod
    def _gaussian_confidence_interval(
            mean: torch.Tensor,
            covariance: torch.Tensor,
            alpha: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get pointwise (diagonal) confidence intervals for a multivariate Gaussian's coordinates.

        If the Gaussian is "multi-task", then a confidence interval is computed for each task.

        :param mean: (d,) or (d, t) The mean of the Gaussian.
        :param covariance: (d,d) or (d*t, d*t) The covariance matrix of the Gaussian.
        :param alpha: The significance of the interval.
        :return: The (``median``, ``lower``, ``upper``) bounds of the confidence interval each of shape (d,) or (d,t)
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
    def _add_jitter(
            distribution: gpytorch.distributions.MultivariateNormal
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Add diagonal jitter to covariance matrices to avoid indefinite covariance matrices.

        :param distribution: The distribution to be jittered.
        :returns: The given distribution with a new covariance matrix including some jitter.
        """
        try:
            covar = distribution.covariance_matrix
        except AttributeError:
            return distribution
        jitter = gpytorch.settings.cholesky_jitter.value(covar.dtype) * 10
        return distribution.add_jitter(jitter)
