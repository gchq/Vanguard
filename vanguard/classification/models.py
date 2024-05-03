"""
Contains model classes to enable classification in Vanguard.
"""
import warnings

import gpytorch
import torch
from gpytorch import settings
from gpytorch.lazy import DiagLazyTensor
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP
from gpytorch.utils.warnings import GPInputWarning

from vanguard.models import ExactGPModel


class DummyKernelDistribution:
    """
    A dummy distribution to hold a kernel matrix and some one-hot labels.
    """
    def __init__(self, labels: torch.Tensor, kernel: torch.Tensor):
        """
        Initialise self.

        :param labels: The one-hot labels.
        :param kernel: The kernel matrix.
        """
        self.labels = labels
        self.kernel = kernel
        self.mean = self.kernel @ self.labels.evaluate()
        self.covariance_matrix = torch.zeros_like(self.mean)

    def add_jitter(self, *args, **kwargs):
        return self


class InertKernelModel(ExactGPModel):
    """
    An inert model wrapping a kernel matrix.

    Uses a given kernel for prior and posterior and returns a dummy distribution holding the
    kernel matrix.
    """
    def __init__(
            self, train_inputs: torch.Tensor, train_targets: torch.Tensor,
            covar_module: gpytorch.kernels.Kernel, mean_module: gpytorch.means.Mean,
            likelihood: gpytorch.likelihoods.Likelihood, num_classes: int
    ):
        """
        Initialise self.

        :param train_inputs: (n_samples, n_features) The training inputs (features).
        :param train_targets: (n_samples,) The training targets (response).
        :param covar_module:  The prior kernel function to use.
        :param mean_module: Not used, remaining in the signature for compatibility.
        :param likelihood:  Likelihood to use with model.
        :param num_classes: The number of classes to use.
        """
        super(ExactGP, self).__init__()

        if train_inputs is None:
            self.train_inputs = None
            self.train_targets = None
        else:
            if torch.is_tensor(train_inputs):
                train_inputs = (train_inputs,)
            try:
                self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
            except AttributeError:
                raise TypeError("Train inputs must be a tensor, or a list/tuple of tensors")
            self.train_targets = train_targets

        self.prediction_strategy = None
        self.n_classes = num_classes
        self.covar_module = covar_module
        self.mean_module = ZeroMean()
        self.likelihood = likelihood

    def train(self, mode: bool = True) -> ExactGPModel:
        """Set to training mode, if data is not None."""
        if mode is True and (self.train_inputs is None or self.train_targets is None):
            raise RuntimeError(
                "train_inputs, train_targets cannot be None in training mode. "
                "Call .eval() for prior predictions, or call .set_train_data() to add training data."
            )
        return super().train(mode)

    def _label_tensor(self, targets: torch.Tensor) -> torch.Tensor:
        return DiagLazyTensor(torch.ones(self.n_classes))[targets.long()]

    def __call__(self, *args, **kwargs) -> DummyKernelDistribution:
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [arg.unsqueeze(-1) if arg.ndimension() == 1 else arg for arg in args]

        input_equals_training_inputs = all(torch.equal(train_input, input)
                                           for train_input, input in zip(train_inputs, inputs))

        if self.training:
            if settings.debug.on() and not input_equals_training_inputs:
                raise RuntimeError("You must train on the training inputs!")
            kernel_matrix = self.covar_module(*inputs)

        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            kernel_matrix = self.covar_module(*args)

        else:
            if settings.debug.on() and input_equals_training_inputs:
                warnings.warn(
                    "The input matches the stored training data. Did you forget to call model.train()?",
                    GPInputWarning,
                )

            kernel_matrix = self.covar_module(*inputs, *train_inputs)

        return DummyKernelDistribution(self._label_tensor(self.train_targets), kernel_matrix)
