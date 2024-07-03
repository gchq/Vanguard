"""
Contains model classes to enable classification in Vanguard.
"""

import warnings
from typing import Any, Optional

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

    def __init__(self, labels: gpytorch.lazy.LazyTensor, kernel: gpytorch.lazy.LazyTensor) -> None:
        """
        Initialise self.

        :param labels: The one-hot labels.
        :param kernel: The kernel matrix.
        """
        self.labels = labels
        self.kernel = kernel
        self.mean = self.kernel @ self.labels.evaluate()
        self.covariance_matrix = torch.zeros_like(self.mean)

    # pylint: disable-next=unused-argument
    def add_jitter(self, *args: Any, **kwargs: Any):
        return self


class InertKernelModel(ExactGPModel):
    """
    An inert model wrapping a kernel matrix.

    Uses a given kernel for prior and posterior and returns a dummy distribution holding the
    kernel matrix.
    """

    def __init__(
        self,
        train_inputs: Optional[torch.Tensor],
        train_targets: Optional[torch.Tensor],
        covar_module: gpytorch.kernels.Kernel,
        mean_module: Optional[gpytorch.means.Mean],
        likelihood: gpytorch.likelihoods.Likelihood,
        num_classes: int,
        **kwargs: Any,
    ) -> None:
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
            except AttributeError as exc:
                raise TypeError("Train inputs must be a tensor, or a list/tuple of tensors") from exc
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

    def __call__(self, *args: Any, **kwargs: Any) -> DummyKernelDistribution:
        # TODO: Why do we accept variable numbers of arguments here? It seems to throw errors if you provide too many
        #  arguments, and the GPyTorch documentation seems very thin here. Also, `kwargs` is ignored entirely.
        # https://github.com/gchq/Vanguard/issues/292
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [arg.unsqueeze(-1) if arg.ndimension() == 1 else arg for arg in args]

        input_equals_training_inputs = all(
            torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)
        )

        if self.training:
            if settings.debug.on() and not input_equals_training_inputs:
                raise RuntimeError("You must train on the training inputs!")
            kernel_matrix = self.covar_module(*inputs)

        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            # TODO: Prior mode evaluation fails due to a shape mismatch, seemingly due to the reference to
            #  train_targets in the return value.
            # https://github.com/gchq/Vanguard/issues/291
            kernel_matrix = self.covar_module(*args)

        else:
            if settings.debug.on() and input_equals_training_inputs:
                warnings.warn(
                    "The input matches the stored training data. Did you forget to call model.train()?",
                    GPInputWarning,
                )

            kernel_matrix = self.covar_module(*inputs, *train_inputs)

        # TODO: This will fail if train_targets is None. (AttributeError: 'NoneType' object has no attribute 'long')
        # https://github.com/gchq/Vanguard/issues/291
        return DummyKernelDistribution(self._label_tensor(self.train_targets), kernel_matrix)
