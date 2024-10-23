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
Contains the MonteCarloPosteriorCollection class.
"""

from collections.abc import Generator
from typing import NoReturn

import torch
from torch import Tensor

from vanguard.base.posteriors.posterior import Posterior
from vanguard.utils import DummyDistribution


class MonteCarloPosteriorCollection(Posterior):
    """
    A collection of posteriors over a set of points.

    Enables fuzzy predictions and confidence intervals for models without any specific method to handle input
    uncertainty.
    Samples are lazily loaded if more are needed for a better prediction.

    :param posterior_generator: A :class:`~vanguard.base.posteriors.Posterior` object defining an infinite generator
        of posteriors.

    .. warning::
        In order to ensure reproducible output for predictions and confidence
        intervals, a cached sample is used.
    """

    INITIAL_NUMBER_OF_SAMPLES: int = 100
    MAX_POSTERIOR_ERRORS_BEFORE_RAISE: int = 100
    """The maximum number of RuntimeErrors that _yield_posteriors will suppress before raising."""

    def __init__(self, posterior_generator: Generator[Posterior, None, None]) -> None:
        """Initialise self."""
        self._posterior_generator = posterior_generator
        self._posteriors_skipped = 0
        distribution = self._create_updated_distribution(self.INITIAL_NUMBER_OF_SAMPLES)
        super().__init__(distribution)
        # _tensor_sample() isn't a method for dummy distributions
        if not isinstance(distribution, DummyDistribution):
            self._cached_samples = self._tensor_sample()

    @property
    def condensed_distribution(self) -> torch.distributions.Distribution:
        """
        Return the condensed distribution.

        Return a representative distribution of the posterior, with 1-dimensional
        mean and 2-dimensional covariance. In this case, return a distribution
        based on the mean and covariance returned by :meth:`_tensor_prediction`.
        """
        mean, covar = self._tensor_prediction()
        return self._add_jitter(self._make_multivariate_normal(mean, covar))

    def sample(self, n_samples: int = 1) -> Tensor:
        """
        Draw independent samples from the posterior.

        :param n_samples: An integer specifying the number of samples to draw.
        """
        new_distribution = self._create_updated_distribution(n_samples)
        return new_distribution.sample()[-n_samples:]

    @classmethod
    def from_mean_and_covariance(cls, mean: torch.Tensor, covariance: torch.Tensor) -> NoReturn:
        """
        Construct from the mean and covariance of a Gaussian.

        :param mean: (d,) or (d, t) The mean of the Gaussian.
        :param covariance: (d, d) or (dt, dt) The covariance matrix of the Gaussian.
        :returns: The multivariate Gaussian distribution for either a single task or multiple tasks, depending on the
                  shape of the args.
        """
        raise NotImplementedError(
            "Constructing a MonteCarloPosteriorCollection from a single mean and covariance of a"
            "Gaussian is not supported."
        )

    def _tensor_prediction(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the prediction as a tensor.

        Prediction is based on an aggregation of predictions.

        :returns: (``means``, ``covar``) where:

            * ``means``: (n_predictions,) The posterior predictive mean,
            * ``covar``: (n_predictions, n_predictions) The posterior predictive covariance matrix.

        """
        predictions = sum(self._cached_samples) / len(self._cached_samples)
        diffs = [(sample - predictions).reshape(-1, 1) for sample in self._cached_samples]
        covar = sum([diff @ diff.T for diff in diffs]) / (len(self._cached_samples) - 1)
        return predictions, covar

    def _tensor_confidence_interval(
        self,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct confidence intervals around mean of predictive posterior.

        :param alpha: The significance level of the CIs.
        :returns: The (``median``, ``lower``, ``upper``) bounds of the confidence interval for the
                    predictive posterior, each of shape (n_predictions,).
        """
        minimum_number_of_samples_needed = self._decide_mc_num_samples(alpha)
        current_number_of_samples = self.distribution.mean.shape[0]
        number_of_additional_samples_needed = minimum_number_of_samples_needed - current_number_of_samples
        if number_of_additional_samples_needed > 0:
            self._update_existing_distribution(number_of_additional_samples_needed)

        quantile_probs = torch.tensor([alpha / 2, 0.5, 1 - alpha / 2])
        lower, median, upper = torch.quantile(self._cached_samples, quantile_probs, dim=0)
        return median, lower, upper

    def _update_existing_distribution(self, n_new_samples: int) -> None:
        """
        Add new samples and update the distribution, also caching new samples.

        :param n_new_samples: An integer defining the number of new samples to be added.
        """
        new_distribution = self._create_updated_distribution(n_new_samples)
        self.distribution = new_distribution
        self._cached_samples = self._tensor_sample()

    def _create_updated_distribution(self, n_new_samples: int) -> torch.distributions.MultivariateNormal:
        """
        Create a new distribution building upon the old one.

        The distribution for this class is dynamic, and depends on the total
        number of samples used. Calling this method will make new samples, and
        return a new distribution built upon the old one.

        :param n_new_samples: The number of new samples to be added.
        :return: The new distribution, with the same class as the existing distribution.
        :raises TypeError: If the posteriors in :class:`self._posterior_generator` have
            different types, or if they do not match the type of the existing distribution.
        """
        try:
            old_distribution_class = type(self.distribution)
        except AttributeError:
            old_distribution_class = None
            means, covars = [], []
        else:
            means, covars = list(self.distribution.mean), list(self.distribution.covariance_matrix)

        new_distribution_classes = set()

        for new_posterior in self._yield_posteriors(n_new_samples):
            new_distribution = new_posterior.distribution
            new_distribution_classes.add(type(new_distribution))
            means.append(new_distribution.mean)
            covars.append(new_distribution.covariance_matrix)

        try:
            (new_distribution_class,) = new_distribution_classes
        except ValueError as exc:
            raise TypeError(f"Posteriors have multiple distribution types: {repr(new_distribution_classes)}.") from exc

        if old_distribution_class is not None and new_distribution_class != old_distribution_class:
            raise TypeError(f"Cannot add {new_distribution_class} types to {old_distribution_class}.")

        new_collective_mean = torch.stack(means)
        new_collective_covar = torch.stack(covars)

        return new_distribution_class(new_collective_mean, new_collective_covar)

    def _yield_posteriors(
        self,
        num_posteriors: int,
    ) -> Generator[Posterior, None, None]:
        """
        Yield a number of posteriors from the infinite generator.

        :param num_posteriors: The number of posteriors to yield.
        """
        num_yielded = 0
        posteriors_skipped_in_a_row = 0
        while num_yielded < num_posteriors:
            try:
                posterior = next(self._posterior_generator)
            except StopIteration:
                msg = (
                    "ran out of samples from the generator! "
                    "MonteCarloPosteriorCollection must be given an infinite generator."
                )
                raise RuntimeError(msg) from None
            try:
                # Pylint false positive
                torch.linalg.cholesky(posterior.distribution.covariance_matrix)  # pylint: disable=not-callable
            except RuntimeError as exc:
                self._posteriors_skipped += 1
                posteriors_skipped_in_a_row += 1
                if posteriors_skipped_in_a_row >= self.MAX_POSTERIOR_ERRORS_BEFORE_RAISE:
                    msg = (
                        f"{posteriors_skipped_in_a_row} errors in a row were caught "
                        f"while generating posteriors - aborting"
                    )
                    raise RuntimeError(msg) from exc
            else:
                yield posterior
                posteriors_skipped_in_a_row = 0
                num_yielded += 1

    def _tensor_log_probability(
        self,
        y: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute the MC approximated log-probability under the posterior.

        :param y: (n, d) or (d,) where d is the dimension of the space on which
            the posterior is defined. Sum over first dimension if two dimensional.
        :returns: The log-likelihood of the given y values, i.e. :math:`\sum_{i} \log P(y_i)` where :math:`P` is the
                  posterior density. :math:`P` is approximated by :math:`P(y) = \frac{1}{N}\sum_{i=1}^N \log P_i(y)`
                   as a sum over a collection of posterior log probabilities.
        """
        log_probs = self.distribution.log_prob(y)
        sum_dimensions = list(range(1, log_probs.ndim))
        if sum_dimensions:
            log_probs = log_probs.sum(dim=sum_dimensions)
        log_prob = torch.logsumexp(log_probs, dim=0)
        log_prob -= torch.log(torch.as_tensor(log_probs.shape[0], dtype=log_prob.dtype, device=log_prob.device))
        return log_prob

    @staticmethod
    def _decide_mc_num_samples(
        alpha: float,
    ) -> int:
        r"""
        Determine an appropriately large number of Monte Carlo samples.

        Determine an appropriately large number of Monte Carlo samples for a desired confidence level when computing
        confidence intervals with Monte Carlo integration. This method is motivated by a simple remark in
        :cite:`Owen13`. The factor is arbitrary, we just want the number of samples to be a lot larger than
        :math:`\frac{1}{\min(alpha, 1-alpha)}`.

        .. warning::
            The current method should give reasonable default behaviour, but it doesn't come with any guarantees.
            Moreover, we may be demanding too many samples, which is inefficient.

        :param alpha: The significance level.
        :return: The number of samples.
        """
        return int(5 / min(alpha / 2, 1 - alpha / 2))
