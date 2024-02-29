"""
Contains a torch distribution implementing a warped Gaussian.
"""
import torch
from torch.distributions import Normal

from ..base.basecontroller import BaseGPController


class WarpedGaussian(Normal):
    r"""
    A warped Gaussian distribution.

    .. math::
        X\sim \mathcal{WN}(\psi; \mu, \sigma) ~ \iff  \psi(X)\sim\mathcal{N}(\mu, \sigma).
    """
    def __init__(self, warp, *args, **kwargs):
        """
        :param `~vanguard.warps.basefunction.WarpFunction warp`: The warp to be used to define the distribution.
        """
        super().__init__(*args, **kwargs)
        self.warp = warp

    def log_prob(self, value):
        """
        Calculate the log-probability of the values under the warped Gaussian distribution.

        :param torch.Tensor value: Shape should be compatible with the distributions shape.
        :returns: The log probability of the values.
        :rtype: torch.Tensor
        """
        gaussian = super().log_prob(self.warp(value))
        jacobian = torch.log(self.warp.deriv(value).abs())
        return gaussian + jacobian

    def sample(self, *args, **kwargs):
        """
        Sample from the distribution.
        """
        gaussian_samples = super().sample(*args, **kwargs)
        return self.warp.inverse(gaussian_samples)

    @classmethod
    def from_data(cls, warp, samples, optimiser=torch.optim.Adam, n_iterations=100, lr=0.001):
        """
        Fit a warped Gaussian distribution to the given data using the supplied warp.

        The mean and variance will be
        optimised along with the free parameters of the warp.

        :param `~vanguard.warps.basefunction.WarpFunction` warp: The warp to use.
        :param array_like[float] samples: (n_samples, ...) The data to fit.
        :param type optimiser: A subclass of class:`torch.optim.Optimizer` used to tune the parameters.
        :param int n_iterations: The number of optimisation iterations.
        :param float lr: The learning rate for optimisation.
        :returns: A fit distribution.
        """
        t_samples = torch.as_tensor(samples, dtype=BaseGPController._default_tensor_type.dtype)
        optim = optimiser(params=[{"params": warp.parameters(), "lr": lr}])

        for i in range(n_iterations):
            loss = -cls._mle_log_prob_parametrised_with_warp_parameters(warp, t_samples)
            loss.backward(retain_graph=i < n_iterations-1)
            optim.step()
        w_samples = warp(t_samples)
        loc = w_samples.mean(dim=0).detach()
        scale = w_samples.std(dim=0).detach() + 1e-4
        distribution = cls(warp, loc=loc, scale=scale)

        return distribution

    @staticmethod
    def _mle_log_prob_parametrised_with_warp_parameters(warp, data):
        """
        Compute the log probability of the data under the warped Gaussian.

        This is done using the optimal MLEs for the Gaussian mean and variance
        parameters, leaving only a function of the warp parameters.
        """
        w_data = warp(data)
        loc = w_data.mean(dim=0).detach()
        scale = w_data.std(dim=0).detach() + 1e-4
        gaussian_log_prob = (-(w_data - loc) ** 2 / (2 * scale ** 2) - torch.log(scale)).sum()
        log_jacobian = torch.log(warp.deriv(data).abs()).sum()
        return gaussian_log_prob + log_jacobian
