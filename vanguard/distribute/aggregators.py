# © Crown Copyright GCHQ
#
# Licensed under the GNU General Public License, version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A suite of aggregators to be used with the :class:`~vanguard.distribute.decorator.Distributed` decorator.

These are responsible for combining the predictions of several independent expert controllers.
"""

from typing import Optional

import torch


class BadPriorVarShapeError(ValueError):
    pass


class BaseAggregator:
    """
    Aggregate experts' posteriors to an approximate predictive posterior.

    All aggregators should inherit from this class.

    :param means: List with `d` elements, each element is an array of a single expert's predictive mean
        at the evaluation points.
    :param covars: List with `d` elements, each :math:`d \times d` element is the individual experts posterior
        predictive covariance at the test points.
    :param prior_var: Tensor with `d` elements, with each element being the diagonal of the test kernel with
        added noise.
    """

    def __init__(
        self, means: list[torch.Tensor], covars: list[torch.Tensor], prior_var: Optional[torch.Tensor] = None
    ) -> None:
        """
        Initialise the BaseAggregator class.
        """
        self.means = torch.stack(means).type(torch.float32)
        self.covars = torch.stack(covars).type(torch.float32)
        self.variances = self.covars.diagonal(dim1=1, dim2=2)
        self.prior_var = torch.as_tensor(prior_var).type(torch.float32) if prior_var is not None else None
        self.n_experts = self.means.shape[0]

        if prior_var is None:
            self.prior_var = None
        else:
            self.prior_var = torch.as_tensor(prior_var).type(torch.float32)
            if self.prior_var.dim() >= self.variances.dim() and self.prior_var.shape != self.variances.shape:
                raise BadPriorVarShapeError(
                    f"Prior var shape {self.prior_var.shape} doesn't match variances shape {self.variances.shape}"
                )

    def aggregate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the predictions of the individual experts into a single PoE prediction.

        :return: The mean and variance of the combined experts.
        """
        # TODO: should this be an abstract method?
        # https://github.com/gchq/Vanguard/issues/241
        raise NotImplementedError

    def _beta_correction(self, delta_diff: torch.Tensor, delta_val: torch.Tensor) -> torch.Tensor:
        """
        Implement the correction to experts' weights.

        .. note::
            Delta is the difference in differential entropy between
            prior and posterior :cite:`Deisenroth15`. ``delta_diff`` and ``delta_val`` are the same in
            :class:`XBCMAggregator`.

        :param delta_diff: The delta used to determine if correction is applied
                (proxy for in-vs-out of training data).
        :param delta_val: The delta value to be corrected. Must be the same shape as ``delta_diff``.
        :return: The corrected expert weights, the same shape as ``delta_diff``.
        """
        in_training_data = delta_diff > 1
        not_in_training_data = delta_diff <= 1
        corrected_weights = in_training_data * delta_val
        corrected_weights += not_in_training_data * (delta_val + ((1 - delta_val) / self.n_experts))
        return corrected_weights

    @staticmethod
    def _make_pseudo_covar(variance: torch.Tensor) -> torch.Tensor:
        """
        Convert a variance to a covariance matrix, where all entries except the diagonal are zeros.

        :param variance: Tensor of variances for each point of interest
        :return: Covariance matrix, where all non-diagonal elements are zero
        """
        dim = variance.size(-1)
        covar = torch.zeros((dim, dim), dtype=variance.dtype)
        covar[range(dim), range(dim)] = variance
        return covar


class POEAggregator(BaseAggregator):
    r"""
    Implements the Product-of-Experts method of :cite:`Deisenroth15`. Formulae for covariances from :cite:`Cao14`.

    Given the posteriors of the experts :math:`p_{i}(y|x) = N(\mu_{i}(x), \sigma_{i}^{2}(x))` for
    :math:`i=1, 2, ..., M`, we define the joint posterior as a Gaussian with moments

    .. math ::
        \mu &= \sigma^{2} \sum_{i} \sigma_{i}^{-2}(x) \mu_{i}(x) \\
        \sigma^{-2} &= \sum_{i} \sigma_{i}^{-2}(x)
    """

    def aggregate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the predictions of the individual experts into a single PoE prediction.

        :return: The mean and variance of the combined experts.
        """
        covar_inverses = torch.stack([torch.inverse(covar) for covar in self.covars])
        covar = torch.inverse(torch.sum(covar_inverses, dim=0))
        mean = torch.tensordot(
            torch.sum(
                torch.stack(
                    [
                        torch.tensordot(mean.reshape(1, -1), covar_inverse, dims=1)
                        for mean, covar_inverse in zip(self.means, covar_inverses)
                    ]
                ),
                dim=0,
            ),
            covar,
            dims=1,
        ).reshape(-1)
        return mean, covar


class EKPOEAggregator(POEAggregator):
    r"""
    Implements a correction to the Product-of-Experts method.

    Given the posteriors of the experts :math:`p_{i}(y|x) = N(\mu_{i}(x), \sigma_{i}^{2}(x))` for
    :math:`i=1, 2, ..., M`, we define the joint posterior as a Gaussian with moments

    .. math ::
        \mu &= M \sigma^{2} \sum_{i} \sigma_{i}^{-2}(x) \mu_{i}(x) \\
        \sigma^{-2} &= \frac{1}{M} \sum_{i} \sigma_{i}^{-2}(x)
    """

    def aggregate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the predictions of the individual experts into a single PoE prediction.

        :return: The mean and variance of the combined experts.
        """
        mean, variance = super().aggregate()
        return mean, variance * self.n_experts


class GPOEAggregator(BaseAggregator):
    r"""
    Implements the Generalised Product-of-Experts method of :cite:`Deisenroth15`.

    Given the posteriors of the experts :math:`p_{i}(y|x) = N(\mu_{i}(x), \sigma_{i}^{2}(x))` for
    :math:`i=1, 2, ..., M`, we define the joint posterior as a Gaussian with moments

    .. math ::
        \mu &= \sigma^{2} \sum_{i} \beta_{i} \sigma_{i}^{-2}(x) \mu_{i}(x) \\
        \sigma^{-2} &= \sum_{i} \beta_{i} \sigma_{i}^{-2}(x)

    where :math:`\beta_{i}=\frac{1}{M}`.
    """

    def aggregate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the predictions of the individual experts into a single PoE prediction.

        :return: The mean and variance of the combined experts.
        """
        beta = torch.ones_like(self.means) / self.n_experts
        mean = torch.sum((beta / self.variances) * self.means, dim=0)
        variance = torch.sum(beta / self.variances, dim=0)
        return mean / variance, self._make_pseudo_covar(1 / variance)


class BCMAggregator(BaseAggregator):
    r"""
    Implements the Bayesian Committee Machine method of :cite:`Deisenroth15`.

    Given the posteriors of the experts :math:`p_{i}(y|x) = N(\mu_{i}(x), \sigma_{i}^{2}(x))` for
    :math:`i=1, 2, ..., M`, we define the joint posterior as a Gaussian with moments

    .. math ::
        \mu &= \sigma^{2} \sum_{i} \sigma_{i}^{-2}(x) \mu_{i}(x) \\
        \sigma^{-2} &= \sum_{i} \sigma_{i}^{-2}(x) + \bigg( 1 - M \bigg) \sigma_{**}^{-2}

    where :math:`\sigma_{**}^{-2}` is the diagonal of the covariance matrix formed by applying the kernel
    on all pairs of points in :math:`x`.
    """

    def aggregate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the predictions of the individual experts into a single BCM prediction.

        :return: The mean and variance of the combined experts.
        """
        beta = torch.ones_like(self.means)
        mean = torch.sum((beta / self.variances) * self.means, dim=0)
        variance = torch.sum(beta / self.variances, dim=0)
        variance -= (self.n_experts - 1) / self.prior_var
        return mean / variance, self._make_pseudo_covar(1 / variance)


class RBCMAggregator(BaseAggregator):
    r"""
    Implements the Robust Bayesian Committee Machine method of :cite:`Deisenroth15`.

    Given the posteriors of the experts :math:`p_{i}(y|x) = N(\mu_{i}(x), \sigma_{i}^{2}(x))` for
    :math:`i=1, 2, ..., M`, we define the joint posterior as a Gaussian with moments

    .. math ::
        \mu &= \sigma^{2} \sum_{i} \sigma_{i}^{-2}(x) \mu_{i}(x) \\
        \sigma^{-2} &= \sum_{i} \sigma_{i}^{-2}(x) + \bigg( 1 - \sum_{i} \beta_{i} \bigg) \sigma_{**}^{-2}

    where :math:`\beta_{i}=0.5(\log \sigma_{*}^{2} - \log \sigma_{i}^{2}(x))` is the difference
    between the prior and the posterior, and :math:`\sigma_{**}^{-2}` is the diagonal of the
    covariance matrix formed by applying the kernel on all pairs of points in :math:`x`.
    """

    def aggregate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the predictions of the individual experts into a single RBCM prediction.

        :return: The mean and variance of the combined experts.
        """
        beta = 0.5 * (torch.log(self.prior_var) - torch.log(self.variances)).reshape(self.n_experts, -1)
        mean = torch.sum((beta / self.variances) * self.means, dim=0)
        variance = torch.sum((beta / self.variances) - (beta / self.prior_var), dim=0) + (1 / self.prior_var)
        return mean / variance, self._make_pseudo_covar(1 / variance)


class XBCMAggregator(BaseAggregator):
    r"""
    Implements the Corrected Bayesian Committee Machine method.

    We define the joint posterior as in :class:`RBCMAggregator`, but with a correction on \beta.
    (For further details see :meth:`BaseAggregator._beta_correction`.)
    """

    def aggregate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the predictions of the individual experts into a single XBCM prediction.

        :return: The mean and variance of the combined experts.
        """
        delta = 0.5 * (torch.log(self.prior_var) - torch.log(self.variances)).reshape(self.n_experts, -1)
        beta = self._beta_correction(delta, delta)
        mean = torch.sum((beta / self.variances) * self.means, dim=0)
        variance = torch.sum((beta / self.variances) - (beta / self.prior_var), dim=0) + (1 / self.prior_var)
        return mean / variance, self._make_pseudo_covar(1 / variance)


class GRBCMAggregator(BaseAggregator):
    r"""
    Implements the Generalised Robust Bayesian Committee Machine method of :cite:`Liu18`.

    Given the posteriors of the experts :math:`p_{i}(y|x) = N(\mu_{i}(x), \sigma_{i}^{2}(x))` for
    :math:`i=1, 2, ..., M`, we define the joint posterior as a Gaussian with moments

    .. math ::
        \mu &= \sigma^{2} \bigg[ \sum_{i=2}^{M} \beta_{i} \sigma_{i}^{-2}(x) \mu_{i}(x)
        + \bigg( 1 - \sum_{i=2}^{M} \beta_{i} \bigg) \sigma_{1}^{-2}(x) \mu_{1}(x) \bigg] \\
        \sigma^{-2} &= \sum_{i=2}^{M} \beta_{i} \sigma_{i}^{-2}(x)
        + \bigg( 1 - \sum_{i=2}^{M} \beta_{i} \bigg) \sigma_{1}^{-2}(x)

    where

    .. math ::
        \beta_{i} = \begin{cases}
            1, & i=2 \\
            0.5(\log \sigma_{1}^{2}(x) - \log \sigma_{i}^{2}(x)), & 3 \leq i \leq M
        \end{cases}
    """

    def aggregate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the predictions of the individual experts into a single GRBCM prediction.

        :return: The mean and variance of the combined experts.
        """
        comm_mean = self.means[0]
        comm_var = self.variances[0]
        means = self.means[1:]
        variances = self.variances[1:]

        beta = 0.5 * (torch.log(comm_var) - torch.log(variances)).reshape(self.n_experts - 1, -1)
        beta[0, :] = 1

        mean = torch.sum((beta / variances) * means - (beta * comm_mean / comm_var), dim=0) + (comm_mean / comm_var)
        variance = torch.sum((beta / variances) - (beta / comm_var), dim=0) + (1 / comm_var)
        return mean / variance, self._make_pseudo_covar(1 / variance)


class XGRBCMAggregator(BaseAggregator):
    r"""
    Implements the Corrected Generalised Robust Bayesian Committee Machine method.

    We define the joint posterior as in :class:`RBCMAggregator`, but with a correction on \beta.
    (For further details see :meth:`BaseAggregator._beta_correction`.)
    """

    def aggregate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the predictions of the individual experts into a single XGRBCM prediction.

        :return: The mean and variance of the combined experts.
        """
        comm_mean = self.means[0]
        comm_var = self.variances[0]
        means = self.means[1:]
        variances = self.variances[1:]

        delta = 0.5 * (torch.log(comm_var) - torch.log(variances)).reshape(self.n_experts - 1, -1)
        delta[0, :] = 1
        delta_diff = 0.5 * (torch.log(self.prior_var) - torch.log(variances)).reshape(self.n_experts - 1, -1)
        beta = self._beta_correction(delta_diff, delta)

        mean = torch.sum((beta / variances) * means - (beta * comm_mean / comm_var), dim=0) + (comm_mean / comm_var)
        variance = torch.sum((beta / variances) - (beta / comm_var), dim=0) + (1 / comm_var)
        return mean / variance, self._make_pseudo_covar(1 / variance)
