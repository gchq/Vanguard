"""
Contains GPyTorch likelihoods required in Vanguard but not implemented in GPyTorch.
"""
from typing import Any, Optional

from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import DiagLazyTensor
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from torch import Size, Tensor


class FixedNoiseMultitaskGaussianLikelihood(MultitaskGaussianLikelihood):
    """
    A multitask likelihood with heteroskedastic noise.

    Combines :class:`gpytorch.likelihoods.MultitaskGaussianLikelihood` with
    :class:`gpytorch.likelihoods.FixedNoiseGaussianLikelihood` to give a multitask Gaussian likelihood
    where a fixed heteroskedastic observation noise can be specified for each training point and task,
    but there is covariance between the points or the tasks.
    """

    def __init__(
        self, noise: Tensor, learn_additional_noise: bool = False, batch_shape: Size = Size(), **kwargs: Any
    ) -> None:
        """
        Initialise self.

        :param noise: (n_samples, n_tasks) The fixed observation noise.
        :param learn_additional_noise: If to learn additional observation (likelihood) noise covariance
            along with the specified fixed noise. Takes the same form as the covariance in
            :class:`gpytorch.likelihoods.MultitaskGaussianLikelihood`.
        :param batch_shape: The batch shape of the learned noise parameter, defaults to empty :class:`~torch.Size`.
        """
        super().__init__(batch_shape=batch_shape, **kwargs)
        self._fixed_noise = noise
        self.learn_additional_noise = learn_additional_noise

    @property
    def fixed_noise(self) -> Tensor:
        """Get the fixed noise."""
        return self._fixed_noise

    @fixed_noise.setter
    def fixed_noise(self, value: Tensor) -> None:
        """Set the fixed noise."""
        self._fixed_noise = value

    def marginal(
        self, function_dist: MultitaskMultivariateNormal, *params: Any, noise: Optional[Tensor] = None, **kwargs: Any
    ) -> MultitaskMultivariateNormal:
        r"""
        Return the marginal distribution.

        If ``rank == 0``, adds the task noises to the diagonal of the covariance matrix of the supplied
        :class:`gpytorch.distributions.MultivariateNormal` or
        :class:`gpytorch.distributions.MultitaskMultivariateNormal`. Otherwise, adds a rank ``rank``
        covariance matrix to it.

        To accomplish this, we form a new :class:`gpytorch.lazy.KroneckerProductLazyTensor` between :math:`I_{n}`,
        an identity matrix with size equal to the data and a (not necessarily diagonal) matrix containing the task
        noises :math:`D_{t}`.

        We also incorporate a shared ``noise`` parameter from the base
        :class:`gpytorch.likelihoods.GaussianLikelihood` that we extend.

        There is also the fixed noise (supplied to
        :meth:`~vanguard.multitask.likelihoods.FixedNoiseMultitaskGaussianLikelihood.__init__`
        as ``noise``) represented as :math:`\sigma^*` of length :math:`nt` with task-contiguous blocks.

        The final covariance matrix after this method is then
        :math:`K + D_{t} \otimes I_{n} + \sigma^{2}I_{nt} + diag(\sigma^*)`.

        :param function_dist: Random variable whose covariance
            matrix is a :class:`gpytorch.lazy.LazyTensor` we intend to augment.
        :param noise: The noise (standard deviation) to use in the likelihood, None, to use the
            likelihoods's own fixed noise.
        :returns: A new random variable whose covariance matrix is a :class:`gpytorch.lazy.LazyTensor` with
            :math:`D_{t} \otimes I_{n}`, :math:`\sigma^{2}I_{nt}` and :math:`diag(\sigma^*)` added.
        """
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        covar_kronecker_lt = self._shaped_noise_covar(mean.shape, add_noise=self.has_global_noise, noise=noise)
        covar = covar + covar_kronecker_lt

        return function_dist.__class__(mean, covar)

    def _shaped_noise_covar(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        base_shape: Size,
        add_noise: bool = True,
        noise: Optional[Tensor] = None,
        *params: Any,
    ) -> DiagLazyTensor:
        """
        Format likelihood noise (i.e. pointwise standard-deviations) as a covariance matrix.

        .. warning::

            The signature of this method is incompatible with base class
            :class:`~gpytorch.likelihoods.multitask_gaussian_likelihood._MultitaskGaussianLikelihoodBase`.

        :param base_shape: The output shape (required for reshaping noise).
        :param add_noise: If to include global additive noise.
        :param noise: Specified noise for the likelihood, or use its own noise if None.
        :returns: Formatted likelihood noise.
        """
        result = DiagLazyTensor(self._flatten_noise(noise if noise is not None else self.fixed_noise))
        if self.learn_additional_noise:
            additional_learned_noise = super()._shaped_noise_covar(base_shape, add_noise=add_noise, *params)
            result += additional_learned_noise
        return result

    @staticmethod
    def _flatten_noise(noise: Tensor) -> Tensor:
        """
        Flatten a noise tensor into a single dimension.

        We encounter covariance matrices in block form where the diagonal blocks are the covariances for the
        individual tasks. We therefore need to convert observation variances of the shape (N, T) to diagonal
        matrices of shape (NT, NT). We wrap it in a convenience function for the sake of readability since
        the transformation is a little unintuitive.

        :param noise: (n_samples, n_tasks) The array of observation variances
            for each tasks' data point.

        :returns: Reshaped 1d tensor. Contiguous within tasks.
        """
        return noise.T.reshape(-1)
