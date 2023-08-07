"""
Contains some multitask classification likelihoods.
"""
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import Positive
from gpytorch.lazy import DiagLazyTensor
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.likelihoods import SoftmaxLikelihood as _SoftmaxLikelihood
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
from gpytorch.likelihoods.noise_models import MultitaskHomoskedasticNoise
import torch

from .models import DummyKernelDistribution


class DummyNoise:
    """
    Provides a dummy wrapper around a tensor so that the tensor can be accessed as the noise property of the class.
    """
    def __init__(self, value):
        """
        Initialise self.

        :param value: Always returned by the :py:attr:noise property.
        """
        self.value = value

    @property
    def noise(self):
        return self.value


class MultitaskBernoulliLikelihood(BernoulliLikelihood):
    """
    A very simple extension of :py:class:`gpytorch.likelihoods.BernoulliLikelihood`.

    Provides an improper likelihood over multiple independent Bernoulli distributions.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialise self and ignore the num_tasks kwarg that may be passed to multi-task likelihoods.
        """
        kwargs.pop("num_classes", None)
        kwargs.pop("num_tasks", None)
        super().__init__(*args, **kwargs)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        """Compute the log probability sum summing the log probabilities over the tasks."""
        return super().log_prob(observations, function_dist, *args, **kwargs).sum(dim=-1)

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        """Compute the expected log probability sum summing the expected log probabilities over the tasks."""
        return super().expected_log_prob(observations, function_dist, *args, **kwargs).sum(dim=-1)


class SoftmaxLikelihood(_SoftmaxLikelihood):
    """
    Superficial wrapper around the GPyTorch :py:class:`gpytorch.likelihoods.SoftmaxLikelihood`.

    This wrapper allows the arg names more consistent with other likelihoods.
    """
    def __init__(self, *args, num_classes=None, num_tasks=None, **kwargs):
        r"""
        Initialise self.

        :param args: For full signature, see :py:class:`gpytorch.likelihoods.SoftmaxLikelihood`.
        :param int,None num_classes: The number of target classes.
        :param int num_tasks: Dimensionality of latent function :math:`\mathbf f`.
        :param kwargs: For full signature, see :py:class:`gpytorch.likelihoods.SoftmaxLikelihood`.
        """
        super().__init__(*args, num_classes=num_classes, num_features=num_tasks, **kwargs)


class DirichletKernelDistribution(torch.distributions.Dirichlet):
    """
    A pseudo Dirichlet distribution with the log probability modified to match that from [CITATION NEEDED]_.
    """
    def __init__(self, label_matrix, kernel_matrix, alpha):
        """
        Initialise self.

        :param torch.Tensor label_matrix: (``n_data_points``,``n_classes``) A binary indicator matrix encoding the class
                                                                            to which each data point belongs.
        :param torch.Tensor kernel_matrix: (``n_data_points``,``n_data_points``) The evaluated kernel matrix.
        :param float alpha: (``n_classes``,) The Dirichlet prior concentration parameters.
        """
        self.label_matrix = label_matrix
        self.kernel_matrix = kernel_matrix
        self.alpha = alpha

        concentration = (self.kernel_matrix @ self.label_matrix + torch.unsqueeze(self.alpha, 0)).evaluate()
        super().__init__(concentration)

    def log_prob(self, value):
        one_hot_values = DiagLazyTensor(torch.ones(self.label_matrix.shape[1]))[value.long()]
        all_class_grouped_kernel_entries = (self.kernel_matrix @ one_hot_values + torch.unsqueeze(self.alpha, 0))
        relevant_logits = all_class_grouped_kernel_entries.evaluate().log() * one_hot_values.evaluate()
        partition_function = (self.alpha.sum() + self.kernel_matrix.sum(dim=-1)).log()
        return relevant_logits.sum() - partition_function.sum()


class DirichletKernelClassifierLikelihood(_OneDimensionalLikelihood):
    """
    A pseudo Dirichlet likelihood matching the approximation in [CITATION NEEDED]_.
    """
    def __init__(self, num_classes, alpha=None, learn_alpha=False, **kwargs):
        """
        Initialise self.

        :param int num_classes: The number of classes in the data.
        :param float,array_like[float],None alpha: The Dirichlet prior concentration. If a float will be assumed
                                                    homogenous.
        :param bool learn_alpha: If to learn the Dirichlet prior concentration as a parameter.
        """
        super().__init__()
        self.n_classes = num_classes
        if alpha is None:
            self._alpha_var = torch.ones(self.n_classes)
        else:
            self._alpha_var = torch.as_tensor(alpha) * torch.ones(self.n_classes)

        if learn_alpha:
            alpha_prior = kwargs.get("alpha_prior", None)
            alpha_constraint = kwargs.get("alpha_constraint", Positive())
            alpha_val = self._alpha_var.clone()
            self._alpha_var = MultitaskHomoskedasticNoise(num_classes, noise_constraint=alpha_constraint,
                                                          noise_prior=alpha_prior)
            self._alpha_var.initialize(noise=alpha_val)
        else:
            self._alpha_var = DummyNoise(self._alpha_var)

    @property
    def alpha(self):
        return self._alpha_var.noise

    def forward(self, function_samples, **kwargs):
        return None

    def log_marginal(self, observations, function_dist, **kwargs):
        marginal = self.marginal(function_dist, **kwargs)
        return marginal.log_prob(observations)

    def marginal(self, function_dist, *args, **kwargs):
        return DirichletKernelDistribution(function_dist.labels, function_dist.kernel,  self.alpha)

    def __call__(self, input, *args, **kwargs):
        is_conditional = torch.is_tensor(input)
        is_marginal = isinstance(input, DummyKernelDistribution)

        if is_conditional:
            return super().__call__(input, *args, **kwargs)
        elif is_marginal:
            return self.marginal(input, *args, **kwargs)
        else:
            raise RuntimeError(
                "Likelihoods expects a DummyKernelDistribution input to make marginal predictions, or a "
                f"torch.Tensor for conditional predictions. Got a {type(input).__name__}"
            )


class GenericExactMarginalLogLikelihood(ExactMarginalLogLikelihood):
    """
    A lightweight modification of :py:class:`gpytorch.mlls.ExactMarginalLogLikelihood`.

    This removes some RuntimeErrors that prevent use with non-Gaussian likelihoods even when it is possible to do so.
    """
    def __init__(self, likelihood, model):
        """
        Initialise self.

        :param gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model.
        :param gpytorch.models.ExactGP model: The exact GP .
        """
        super(ExactMarginalLogLikelihood, self).__init__(likelihood, model)

    def forward(self, function_dist, target, *params):
        r"""
        Compute the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        output = self.likelihood(function_dist, *params)
        log_prob_of_marginal = output.log_prob(target)
        res = self._add_other_terms(log_prob_of_marginal, params)

        num_data = target.size(-1)
        scaled_data = res.div_(num_data)
        return scaled_data
