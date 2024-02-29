"""
Gaussian processes can be trained on inputs with uncertainty.
"""
from itertools import islice
import warnings

import gpytorch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch

from .base import GPController
from .optimise import NoImprovementError, SmartOptimiser
from .utils import generator_append_constant, infinite_tensor_generator
from .warnings import _INPUT_WARNING, GPInputWarning


class GaussianUncertaintyGPController(GPController):
    """
    Allows the user to pass the standard deviation of the input values.

    Base class for implementing the HNIGP of [CITATION NEEDED]_. This is a generalised version of the NIGP method in
    :cite:`Mchutchon11` and our implementation here exploits mod:`torch.autograd` to circumvent any by-hand
    calculations of GP derivatives.
    """
    def __init__(self, train_x, train_x_std, train_y, y_std, kernel_class, mean_class=gpytorch.means.ConstantMean,
                 likelihood_class=FixedNoiseGaussianLikelihood,
                 marginal_log_likelihood_class=ExactMarginalLogLikelihood, optimiser_class=torch.optim.Adam,
                 smart_optimiser_class=SmartOptimiser, **kwargs):
        """
        Initialise self.

        :param array_like[float] train_x: (n_samples, n_features) The mean of the inputs (or the observed values)
        :param array_like[float],float,None train_x_std: The standard deviation of input noise:

            * *array_like[float]* (n_features,): observation std dev per input feature,
            * *array_like[float]* (n_samples, n_features): observation std dev input samples in each input dimension,
            * *float*: single input feature, or assumed homoskedastic noise amongst inputs,
            * None: heteroskedastic (among input dimension) std dev is inferred from given noisy inputs.

        :param array_like[float] train_y: (n_samples,) or (n_samples, 1) The responsive values.
        :param type kernel_class: An uninstantiated subclass of class:`gpytorch.kernels.Kernel`.
        :param type mean_class: An uninstantiated subclass of class:`gpytorch.means.Mean` to use in the prior GP.
                Defaults to class:`gpytorch.means.ConstantMean`.
        :param array_like[float],float y_std: The observation noise standard deviation:

            * *array_like[float]* (n_samples,): known heteroskedastic noise,
            * *float*: known homoskedastic noise assumed.

        :param type gp_model_class: An uninstantiated subclass of a GP model from mod:`gpytorch.models`.
                The default is class:`vanguard.models.ExactGPModel`.
        :param type likelihood_class: An uninstantiated subclass of class:`gpytorch.likelihoods.Likelihood`.
                The default is class:`gpytorch.likelihoods.FixedNoiseGaussianLikelihood`.
        :param type marginal_log_likelihood_class: An uninstantiated subclass of of an MLL from
                mod:`gpytorch.mlls`. The default is class:`gpytorch.mlls.ExactMarginalLogLikelihood`.
        :param type optimiser_class: An uninstantiated class:`torch.optim.Optimizer` class used for
                gradient-based learning of hyperparameters. The default is class:`torch.optim.Adam`.
        :param kwargs: For a complete list, see class:`~vanguard.base.gpcontroller.GPController`.
        """
        super().__init__(train_x=train_x, train_y=train_y, kernel_class=kernel_class, mean_class=mean_class,
                         y_std=y_std, likelihood_class=likelihood_class,
                         marginal_log_likelihood_class=marginal_log_likelihood_class, optimiser_class=optimiser_class,
                         smart_optimiser_class=smart_optimiser_class, **kwargs)

        self._gradient_variance = None

        self._learn_input_noise = (train_x_std is None)
        self.train_x_std = self._process_x_std(train_x_std)
        if self.batch_size is None or self._learn_input_noise:
            self.train_x_std = self.train_x_std.to(self.device)

        if self.train_x_std.shape == self.train_x.shape:
            self.train_data_generator = infinite_tensor_generator(self.batch_size, self.device,
                                                                  (self.train_x, 0),
                                                                  (self.train_y, self._y_batch_axis),
                                                                  (self._y_variance, self._y_batch_axis),
                                                                  (self.train_x_std, 0))
        else:
            self.train_data_generator = generator_append_constant(self.train_data_generator,
                                                                  self.train_x_std.to(self.device))

    @property
    def gradient_variance(self):
        r"""
        Return the gradient variance.

        Access the value that stores the :math:`n\times n` tensor of covariance resulting from
        Taylor expansion added to training covariance matrix.
        """
        return self._gradient_variance

    @gradient_variance.setter
    def gradient_variance(self, value):
        """
        Set the gradient variance.

        Update the posterior-mean-gradient additive covariance term and also update the
        fixed noise inside the likelihood.
        """
        self._gradient_variance = value
        self.likelihood_noise = self._original_y_variance_as_tensor + self._noise_transform(value)

    def predict_at_point(self, x):
        """Doesn't make sense for an uncertain controller."""
        raise TypeError("Cannot call 'predict_at_point' directly, try 'predict_at_fuzzy_point'.")

    def _sgd_round(self, n_iters=100, gradient_every=100):
        """
        Use gradient based optimiser to tune the hyperparameters.

        Additive gradient noise is set before each call to the super method.

        :param int n_iters: The number of gradient updates.
        :param int gradient_every: How often (in iterations) to do HNIGP input gradient steps.
        :return: The training loss at the last iteration.
        :rtype: float
        """
        loss, detached_loss = torch.tensor(float("nan"), dtype=self.dtype, device=self.device), float("nan")
        self.set_to_training_mode()

        for iter_num, (train_x, train_y, train_y_noise, train_x_std) in enumerate(islice(self.train_data_generator,
                                                                                         n_iters)):
            if (iter_num + 1) % gradient_every == 0:
                _, _, grad_var_term = self._get_additive_grad_noise(train_x, train_x_std ** 2)
                self._original_y_variance_as_tensor = train_y_noise
                self.gradient_variance = grad_var_term
                self.set_to_training_mode()
            try:
                loss = self._single_optimisation_step(train_x, train_y, retain_graph=(iter_num < n_iters - 1))
            except NoImprovementError:
                loss = self._smart_optimiser.last_n_losses[-1]
                break
            except RuntimeError as err:
                warnings.warn(f"Hit a numerical error after {iter_num} iterations of training.")
                if self.auto_restart:
                    warnings.warn(f"Re-running training from scratch for {iter_num-1} iterations.")
                    self._smart_optimiser.reset()
                    self.metrics_tracker.reset()
                    self._sgd_round(iter_num-1, gradient_every)
                    break
                else:
                    raise RuntimeError("Pass auto_restart=True to the controller to automatically restart training up "
                                       "to the last stable iterations.") from err
            finally:
                try:
                    detached_loss = loss.detach().cpu().item()
                except AttributeError:
                    detached_loss = loss
                self._metrics_tracker.run_metrics(detached_loss, self)
        self._smart_optimiser.set_parameters()
        return detached_loss

    def _set_requires_grad(self, value):
        """Set the requires grad flag of all trainable params."""
        super()._set_requires_grad(value)
        if self._learn_input_noise:
            self.train_x_std.requires_grad = value

    def _get_additive_grad_noise(self, x, x_var):
        """
        Use mod:`torch.autograd` to find the gradient of the posterior mean and derived additive covariance term.

        :param torch.Tensor x: (n_samples, self.dim) The input samples at which to compute the gradient.
        :param torch.Tensor x_var: Input dimension variances:

            * (self.dim,): Input dimension variances,
            * (n_samples, self.dim): Input dimension variances for each input point.

        :returns: (``mean``, ``covar``, ``root_covar``), where:

            * ``mean``: (n_samples,) The posterior predictive mean,
            * ``covar``: (n_samples, n_samples) The posterior predictive covariance matrix,
            * ``root_covar``: (n_samples, n_samples) The additive covariance term from the posterior mean gradient.

        :rtype: tuple[torch.Tensor]
        """
        # Turn gradients off for model trainable params and on for the inputs
        x_with_grad = x.detach().clone()
        x_with_grad.requires_grad = True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=GPInputWarning, message=_INPUT_WARNING)
            posterior = self._get_posterior_over_point_in_eval_mode(x_with_grad)

        preds, covar = posterior._tensor_prediction()

        # Each entry of preds depends only on the matching input vector, so summing them is a simple way of getting the
        # result we need from autograd (which can only compute gradients of scalars)
        preds_grad = []
        if len(preds.shape) == 1:
            preds = preds.unsqueeze(-1)
        sum_over_inputs_preds = preds.sum(dim=0)
        for sum_preds in sum_over_inputs_preds:
            x_with_grad.retain_grad()
            sum_preds.backward(retain_graph=True)
            preds_grad.append(x_with_grad.grad)
        root_var = torch.stack([torch.mul(pg, torch.sqrt(x_var)) for pg in preds_grad])
        return preds.detach(), covar.detach(), root_var

    def _get_posterior_over_fuzzy_point_in_eval_mode(self, x, x_std):
        """
        Obtain posterior predictive mean and covariance at a point with variance.

        .. warning:
            The ``n_features`` must match with attr:`self.dim`.

        :param array_like[float] x: (n_preds, n_features) The predictive inputs.
        :param array_like[float],float x_std: The input noise standard deviations:

            * array_like[float]: (n_features,) The standard deviation per input dimension for the predictions,
            * float: Assume homoskedastic noise.

        :returns: The prior distribution.
        :rtype: vanguard.base.posteriors.Posterior
        """
        tx = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        tx_std = self._process_x_std(x_std).to(self.device)
        preds, covar, additive_grad_noise = self._get_additive_grad_noise(tx, tx_std ** 2)
        additional_covar = torch.diag(self._noise_transform(additive_grad_noise).T.reshape(-1))
        covar += additional_covar

        jitter = torch.eye(covar.shape[0]) * gpytorch.settings.cholesky_jitter.value(covar.dtype)

        return self.posterior_class.from_mean_and_covariance(preds.squeeze(), covar + jitter)

    def _process_x_std(self, std):
        """
        Parse supplied std dev for input noise for different cases.

        :param array_like[float],float,None std: The standard deviation:

            * array_like[float]: (n_point, self.dim) heteroskedastic input noise across feature dimensions,
            * float: homoskedastic input noise across feature dimensions,
            * None: unknown input noise.

        :return: The parsed standard deviation of shape (self.dim,) or (std.shape[0], self.dim) depending on
                    the shape of ``std``. If ``std`` is None then trainable values are returned.
        :rtype: torch.Tensor
        """
        if std is not None:
            std_tensor = super()._process_x_std(std)
        else:
            # Create a learnable tensor of stds, one for each input dimension
            std_tensor = torch.ones(self.dim, dtype=self.dtype) * 0.01
            std_tensor.requires_grad = True
        return std_tensor

    @staticmethod
    def _noise_transform(gamma):
        return torch.stack([torch.diag(torch.matmul(g, g.T)) for g in gamma], -1).squeeze()
