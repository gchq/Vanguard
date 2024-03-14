"""
The (non-user-facing) base class of Vanguard controllers.

The :py:class:`~vanguard.base.basecontroller.BaseGPController` class contains the
machinery of the :py:class:`~vanguard.base.gpcontroller.GPController`.
"""
from __future__ import annotations

from itertools import islice
import warnings

import gpytorch
from gpytorch import constraints
from gpytorch.utils.errors import NanError
import torch
from typing import Callable, Generator, Type, Union
from numpy.typing import ArrayLike
from numpy import dtype

from . import metrics
from ..decoratorutils import wraps_class
from ..models import ExactGPModel
from ..optimise import NoImprovementError, SmartOptimiser
from ..utils import infinite_tensor_generator, instantiate_with_subset_of_kwargs
from ..warnings import _CHOLESKY_WARNING, _JITTER_WARNING, NumericalWarning
from .posteriors import MonteCarloPosteriorCollection, Posterior
from .standardise import StandardiseXModule

NOISE_LOWER_BOUND = 1e-3
ttypes = Type[Union[torch.FloatTensor, torch.DoubleTensor, torch.IntTensor, torch.BoolTensor,
              torch.HalfTensor, torch.BFloat16Tensor, torch.ByteTensor, torch.CharTensor,
              torch.ShortTensor, torch.LongTensor]]
ttypes_cuda = Type[Union[torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.IntTensor,
                   torch.cuda.BoolTensor, torch.cuda.HalfTensor, torch.cuda.BFloat16Tensor,
                   torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                   torch.cuda.LongTensor]]


class BaseGPController:
    """
    Contains the base machinery for the :py:class:`~vanguard.base.gpcontroller.GPController` class.

    :param train_x: (n_samples, n_features) The mean of the inputs (or the observed values)
    :param train_y: (n_samples,) or (n_samples, 1) The responsive values.
    :param kernel_class: An uninstantiated subclass of :py:class:`gpytorch.kernels.Kernel`.
    :param mean_class: An uninstantiated subclass of :py:class:`gpytorch.means.Mean` to use in the prior GP.
    :param y_std: The observation noise standard deviation:

        * *ArrayLike[float]* (n_samples,): known heteroskedastic noise,
        * *float*: known homoskedastic noise assumed.

    :param likelihood_class: An uninstantiated subclass of :py:class:`gpytorch.likelihoods.Likelihood`.
            The default is :py:class:`gpytorch.likelihoods.FixedNoiseGaussianLikelihood`.
    :param marginal_log_likelihood_class: An uninstantiated subclass of an MLL from
            :py:mod:`gpytorch.mlls`. The default is :py:class:`gpytorch.mlls.ExactMarginalLogLikelihood`.
    :param optimiser_class: An uninstantiated :py:class:`torch.optim.Optimizer` class used for
            gradient-based learning of hyperparameters. The default is :py:class:`torch.optim.Adam`.

    :Keyword Arguments:

        * **kernel_kwargs** *(dict)*: Keyword arguments to be passed to the kernel_class constructor.
        * **mean_kwargs** *(dict)*: Keyword arguments to be passed to the mean_class constructor.
        * **likelihood_kwargs** *(dict)*: Keyword arguments to be passed to the likelihood_class constructor.
        * **gp_kwargs** *(dict)*: Keyword arguments to be passed to the gp_model_class constructor.
        * **mll_kwargs** *(dict)*: Keyword arguments to be passed to the
          marginal_log_likelihood_class constructor.
        * **optim_kwargs** *(dict)*: Keyword arguments to be passed to the optimiser_class constructor.
        * **batch_size** *(int,None)*: The batch size to use in SGD. If ``None``, the whole dataset is
          used at each iteration.
        * **additional_metrics** *(List[function])*: A list of additional metrics to track.


    """
    if torch.cuda.is_available():
        _default_tensor_type: ttypes_cuda = torch.cuda.FloatTensor
    else:
        _default_tensor_type: ttypes = torch.FloatTensor

    torch.set_default_tensor_type(_default_tensor_type)

    gp_model_class = ExactGPModel
    posterior_class = Posterior
    posterior_collection_class = MonteCarloPosteriorCollection

    _y_batch_axis: int = 0

    def __init__(
            self,
            train_x: ArrayLike[float],
            train_y: ArrayLike[float],
            kernel_class: Type[gpytorch.kernels.Kernel],
            mean_class: Type[gpytorch.means.Mean],
            y_std: ArrayLike[float],
            likelihood_class: Type[gpytorch.likelihoods.Likelihood],
            marginal_log_likelihood_class: Type[gpytorch.mlls.marginal_log_likelihood.MarginalLogLikelihood],
            optimiser_class: Type[torch.optim.Optimizer],
            smart_optimiser_class: Type[SmartOptimiser],
            **kwargs
    ):
        """Initialise self."""
        if train_x.ndim == 1:
            self.train_x = torch.tensor(train_x, dtype=self.dtype, device="cpu").unsqueeze(1)
        else:
            self.train_x = torch.tensor(train_x, dtype=self.dtype, device="cpu")

        if train_y.ndim == 1:
            self.train_y = torch.tensor(train_y, dtype=self.dtype, device="cpu").unsqueeze(1)
        else:
            self.train_y = torch.tensor(train_y, dtype=self.dtype, device="cpu")

        self.N, self.dim, *_ = self.train_x.shape

        self._original_y_variance_as_tensor = torch.as_tensor(y_std ** 2, dtype=self.dtype)
        if isinstance(y_std, (float, int)):
            self._y_variance = torch.ones_like(self.train_y, dtype=self.dtype).squeeze(dim=-1)\
                               * (y_std ** 2)
        else:
            self._y_variance = torch.as_tensor(y_std ** 2, dtype=self.dtype)

        self.batch_size = kwargs.get("batch_size", None)
        if self.batch_size is None:
            self.train_x = self.train_x.to(self.device)
            self.train_y = self.train_y.to(self.device)
            self._y_variance = self._y_variance.to(self.device)

        all_likelihood_params_as_kwargs = {"noise": self._y_variance,
                                           "noise_constraint": constraints.GreaterThan(NOISE_LOWER_BOUND)}
        all_likelihood_params_as_kwargs.update(kwargs.get("likelihood_kwargs", {}))
        self.likelihood = instantiate_with_subset_of_kwargs(likelihood_class, **all_likelihood_params_as_kwargs)

        mean_class, kernel_class = self._input_standardise_modules(mean_class, kernel_class)

        kernel_kwargs = kwargs.get("kernel_kwargs", {})
        mean_kwargs = kwargs.get("mean_kwargs", {})

        self.kernel = kernel_class(**kernel_kwargs)
        self.mean = mean_class(**mean_kwargs)

        gp_kwargs = kwargs.get("gp_kwargs", {})

        @_catch_and_check_module_errors(controller=self)
        class SafeGPModelClass(self.gp_model_class):
            pass

        @_catch_and_check_module_errors(controller=self)
        class SafeMarginalLogLikelihoodClass(marginal_log_likelihood_class):
            pass

        self._gp = SafeGPModelClass(self.train_x, self.train_y.squeeze(dim=-1), covar_module=self.kernel,
                                    likelihood=self.likelihood, mean_module=self.mean, **gp_kwargs)

        mll_kwargs = kwargs.get("mll_kwargs", {})
        self._mll = SafeMarginalLogLikelihoodClass(self.likelihood, self._gp, **mll_kwargs)

        optimiser_kwargs = kwargs.get("optim_kwargs", {})
        self._smart_optimiser = smart_optimiser_class(optimiser_class, self._gp, **optimiser_kwargs)

        self.train_data_generator = infinite_tensor_generator(self.batch_size, self.device,
                                                              (self.train_x, 0),
                                                              (self.train_y, self._y_batch_axis),
                                                              (self._y_variance, self._y_batch_axis))

        additional_metrics = kwargs.get("additional_metrics", [])
        self._metrics_tracker = metrics.MetricsTracker(metrics.loss, *additional_metrics)

        self.auto_restart = kwargs.get("auto_restart", None)

        self.warn_normalise_y()

    @property
    def dtype(self) -> dtype | None:
        """Return the default dtype of the controller."""
        return self._default_tensor_type.dtype

    @property
    def device(self) -> torch.device:
        """Return the default device of the controller."""
        if self._default_tensor_type.is_cuda:
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    @property
    def _likelihood(self) -> gpytorch.likelihoods.Likelihood:
        """Return the likelihood of the model."""
        return self._gp.likelihood

    def set_to_training_mode(self) -> None:
        """Set trainable parameters to training mode."""
        self._gp.train()
        self._set_requires_grad(True)

    def set_to_evaluation_mode(self) -> None:
        """Set trainable parameters to evaluation mode."""
        self._gp.eval()
        self._set_requires_grad(False)

    def _predictive_likelihood(
            self,
            x: ArrayLike[float],
    ) -> Posterior:
        """
        Calculate the predictive likelihood at an x-value.

        .. warning:
            We assume either a homoskedastic noise model, or a pre-specified noise level via the y_std arg.

        :param x: (n_preds, n_features) The points at which to obtain the likelihood.
        :returns: The marginal distribution.
        """
        posterior = self._get_posterior_over_point_in_eval_mode(x)
        try:
            assumed_homoskedastic_noise = self.likelihood.noise[0]
            noise = assumed_homoskedastic_noise
        except AttributeError:
            shape = self._decide_noise_shape(posterior, x)
            noise = torch.zeros(shape, dtype=self.dtype, device=self.device)
        output = self._likelihood(posterior.distribution, noise=noise)
        return self.posterior_class(output)

    def _fuzzy_predictive_likelihood(
            self,
            x: ArrayLike[float],
            x_std: ArrayLike[float],
    ) -> Posterior:
        """
        Calculate the predictive likelihood at an x-value, given variance.

        :param x: (n_preds, n_features) The points at which to obtain the likelihood.
        :param x_std: (n_preds, n_features) The std-dev of input points.
        :returns: The marginal distribution.
        """
        prediction_output = self._get_posterior_over_fuzzy_point_in_eval_mode(x, x_std)
        shape = self._decide_noise_shape(prediction_output, x)
        noise = torch.zeros(shape, dtype=self.dtype, device=self.device)
        output = self._likelihood(prediction_output.condensed_distribution, noise=noise)
        return self.posterior_class(output)

    def _get_posterior_over_fuzzy_point_in_eval_mode(
            self,
            x: ArrayLike[float],
            x_std: ArrayLike[float],
    ) -> Posterior:
        """
        Obtain Monte Carlo integration samples from the predictive posterior with Gaussian input noise.

        .. warning:
            The ``n_features`` must match with :py:attr:`self.dim`.

        :param x: (n_preds, n_features) The predictive inputs.
        :param x_std: The input noise standard deviations:

            * array_like[float]: (n_features,) The standard deviation per input dimension for the predictions,
            * float: Assume homoskedastic noise.

        :returns: The prior distribution.
        """
        tx = torch.tensor(x, dtype=self.dtype, device=self.device)
        tx_std = self._process_x_std(x_std).to(self.device)

        def infinite_x_samples(
                group_size: int = 100,
        ) -> Generator[torch.Tensor, None, None]:
            """
            Yield infinitely many samples.

            :param group_size: The group size for sampling.
            """
            while True:
                sample_shape = torch.Size([group_size]) + tx.shape
                group_of_samples = torch.randn(size=sample_shape) * tx_std + tx
                yield from group_of_samples

        posteriors = (self._get_posterior_over_point_in_eval_mode(x_sample) for x_sample in infinite_x_samples())
        posterior_collection = self.posterior_collection_class(posteriors)
        return posterior_collection

    def _set_requires_grad(
            self,
            value: bool,
    ) -> None:
        """
        Set the required grad flag of all trainable params.

        :param value: value to set for requires_grad attribute
        """
        for param in self._smart_optimiser.parameters():
            param.requires_grad = value

    def _sgd_round(
            self,
            n_iters: int = 10,
            gradient_every: int = 10,
    ) -> torch.Tensor:
        """
        Use gradient based optimiser to tune the hyperparameters.

        :param n_iters: The number of gradient updates.
        :param gradient_every: How often (in iterations) to do HNIGP input gradient steps.
        :return: The training loss at the last iteration.
        """
        self.set_to_training_mode()
        loss, detached_loss = torch.tensor(float("nan"), dtype=self.dtype, device=self.device), float("nan")

        for iter_num, (train_x, train_y, train_y_noise) in enumerate(islice(self.train_data_generator, n_iters)):
            self.likelihood_noise = train_y_noise
            try:
                loss = self._single_optimisation_step(train_x, train_y, retain_graph=(iter_num < n_iters - 1))

            except NoImprovementError:
                loss = self._smart_optimiser.last_n_losses[-1]
                break
            except RuntimeError as err:
                warnings.warn(f"Hit a numerical error after {iter_num} iterations of training.")
                if self.auto_restart is True:
                    warnings.warn(f"Re-running training from scratch for {iter_num-1} iterations.")
                    self._smart_optimiser.reset()
                    self._sgd_round(iter_num-1, gradient_every)
                else:
                    if self.auto_restart is None:
                        warnings.warn("Pass auto_restart=True to the controller to automatically restart"
                                      " training up to the last stable iterations.")
                    raise err
            finally:
                try:
                    detached_loss = loss.detach().cpu().item()
                except AttributeError:
                    detached_loss = loss
                self._metrics_tracker.run_metrics(detached_loss, self)
        self._smart_optimiser.set_parameters()
        return detached_loss

    def _single_optimisation_step(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        Take do a single forward pass and optimisation backward pass.

        :param x: (n_samples, n_features) The inputs.
        :param y: (n_samples, ?) The response values.
        :returns: The loss.
        """
        self._smart_optimiser.zero_grad()
        loss = self._loss(x, y)
        loss.backward(retain_graph=retain_graph)
        self._smart_optimiser.step(loss, closure=lambda: self._loss(x, y))
        return loss

    def _loss(
            self,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the training loss (negative marginal log likelihood).

        :param train_x: The observed values.
        :param train_y: The responsive values.
        :returns: The loss.
        """
        output = self._gp_forward(train_x)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NumericalWarning, message=_JITTER_WARNING)
            warnings.filterwarnings("ignore", category=NumericalWarning, message=_CHOLESKY_WARNING)
            return -self._mll(output, train_y.squeeze(dim=-1))

    def _get_posterior_over_point_in_eval_mode(
            self,
            x: ArrayLike[float],
    ) -> Posterior:
        """
        Predict the y-value of a single point in evaluation mode.

        :param x: (n_preds, n_features) The predictive inputs.
        :returns: The prior distribution.
        """
        self.set_to_evaluation_mode()
        return self._get_posterior_over_point(x)

    def _gp_forward(
            self,
            x: ArrayLike[float],
    ) -> ExactGPModel:
        """Pass inputs through the base GPyTorch GP model."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NumericalWarning, message=_JITTER_WARNING)
            try:
                with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
                    output = self._gp(x)
            except (AttributeError, RuntimeError):
                output = self._gp(x)
        return output

    def _get_posterior_over_point(
            self,
            x: ArrayLike[float]
    ) -> Posterior:
        """
        Predict the y-value of a single point. The mode (eval vs train) of the model is not changed.

        :param x: (n_preds, n_features) The predictive inputs.
        :returns: The prior distribution.
        """
        tx = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        output = self._gp_forward(tx)
        return self.posterior_class(output)

    def _process_x_std(
            self,
            std: ArrayLike[float],
    ) -> torch.Tensor:
        """
        Parse supplied std dev for input noise for different cases.

        :param array_like[float],float,None std: The standard deviation:

            * array_like[float]: (n_point, self.dim) heteroskedastic input noise across feature dimensions,
            * float: homoskedastic input noise across feature dimensions,

        :return: The parsed standard deviation of shape (self.dim,) or (std.shape[0], self.dim) depending on
                    the shape of ``std``. If ``std`` is ``None`` then trainable values are returned.
        """
        std_tensor = torch.as_tensor(std, dtype=self.dtype, device="cpu")
        if std_tensor.dim() == 0:
            std_tensor = torch.ones(self.dim, dtype=self.dtype, device=self.device) * std_tensor
        return std_tensor

    def _input_standardise_modules(
            self,
            *modules: torch.nn.Module,
    ) -> list[StandardiseXModule]:
        """
        Apply standard input scaling (mean zero, variance 1) to the supplied PyTorch nn.Modules.

        The mean and variance are computed from the training inputs of self.

        :param *modules: Modules to apply mean and variance to.
        """
        norm_module = StandardiseXModule.from_data(self.train_x, device=self.device, dtype=self.dtype)
        scaled_modules = [norm_module.apply(module) for module in modules]
        return scaled_modules

    @classmethod
    def set_default_tensor_type(
            cls,
            tensor_type: ttypes | ttypes_cuda,
    ) -> None:
        """
        Set the default tensor type for the class, subsequent subclasses, and external tensors.

        :param tensor_type: The tensor type to apply as the default
        """
        cls._default_tensor_type = tensor_type
        torch.set_default_tensor_type(tensor_type)

    @staticmethod
    def _decide_noise_shape(
            posterior: Posterior,
            x: torch.Tensor,
    ) -> tuple[int]:
        """
        Determine the correct shape of the likelihood noise.

        Given a posterior distribution and an array of predictive inputs,
        determine the correct size of a noise term to supply to the likelihood
        to match the model and the number of predictive points.

        :param posterior: The posterior distribution that will combined
            with the noise in a likelihood.
        :param x: The predictive input points.
        :returns: The correct shape for the likelihood noise in this case.
        """
        mean, covar = posterior.distribution.mean, posterior.distribution.covariance_matrix

        shape_mapping = {
            (1, 2): (x.shape[0],),  # single MultivariateNormal
            (2, 2): (x.shape[0], mean.shape[-1]),  # single MultitaskMultivariateNormal
            (2, 3): (x.shape[0],),  # batch of MultivariateNormals
            (3, 3): (x.shape[0], mean.shape[-1]),  # batch of MultitaskMultivariateNormals
        }

        try:
            shape = shape_mapping[(mean.ndim, covar.ndim)]
        except KeyError:
            raise ValueError(f"A posterior distribution with mean and covariance matrix of dimensions {mean.ndim} and "
                             f"{covar.ndim} are not currently supported.")
        return shape

    @staticmethod
    def warn_normalise_y() -> None:
        """
        Give a warning to indicate that y values have not been standard scaled.

        Can be overridden and disabled by subclasses when not relevant.
        """
        warnings.warn("A regression problem with no warping may suffer from numerical "
                      "instability in optimisation if the y values are not standard "
                      "scaled. Using the NormaliseY decorator will likely help.")


def _catch_and_check_module_errors(
        controller: BaseGPController,
) -> Callable:
    """
    Handle some hard to detect errors that may occur within GP model classes.

    :param controller: The controller that owns the module class.
    """
    def decorator(
            module_class: torch.nn.Module,
    ) -> torch.nn.Module:
        """
        Decorate a particular module (mean/kernel).

        :param torch.nn.Module module_class: The model class to which to apply error handling.
        """
        @wraps_class(module_class)
        class InnerClass(module_class):
            """
            A safe dynamic subclass of the module class.

            This class overloads the __call__ method to give a more reasonable
            error message when the input data is not rank-1.
            """
            def __call__(self, *args, **kwargs):
                try:
                    result = super().__call__(*args, **kwargs)
                except NanError:  # otherwise we catch this as a RuntimeError
                    raise
                except RuntimeError:
                    decorator_names = {decorator.__name__ for decorator in controller.__decorators__}
                    if controller.train_x.ndim > 2 and "HigherRankFeatures" not in decorator_names:
                        raise ValueError("Input data looks like it might not be rank-1, "
                                         f"shape={str(tuple(controller.train_x.shape))}. If your "
                                         "features are higher rank (e.g. rank-2 for time series) "
                                         "consider using the HigherRankFeatures decorator on your controller "
                                         "and make sure that your kernel and mean functions are defined "
                                         "for the rank of your input features.")
                    else:
                        raise
                else:
                    return result
        return InnerClass
    return decorator
