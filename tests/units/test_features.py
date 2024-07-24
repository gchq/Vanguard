"""
Tests for the HigherRankFeatures decorator.

.. note::
    ``features.py`` or this test module should be refactored if another feature is added.
"""

import unittest
from typing import Any, Type

import gpytorch
import torch
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.means import ConstantMean
from typing_extensions import Self

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import HigherRankSyntheticDataset
from vanguard.features import HigherRankFeatures
from vanguard.kernels import ScaledRBFKernel
from vanguard.standardise import DisableStandardScaling
from vanguard.vanilla import GaussianGPController


class TwoDimensionalLazyEvaluatedKernelTensor(LazyEvaluatedKernelTensor):
    """
    Define an example lazy kernel tensor that works in two dimensions.
    """

    # pylint: disable=abstract-method
    @classmethod
    def from_lazy_evaluated_kernel_tensor(cls: Type[Self], lazy_tensor: gpytorch.lazy.LazyTensor) -> Self:
        """
        Create an instance of the class from a lazy tensor.

        :param lazy_tensor: The lazy tensor to generate the class from.
        :return: Instance of `TwoDimensionalLazyEvaluatedKernelTensor`
        """
        kernel = lazy_tensor.kernel
        x1 = lazy_tensor.x1
        x2 = lazy_tensor.x2
        last_dim_is_batch = lazy_tensor.last_dim_is_batch
        params = lazy_tensor.params
        return cls(x1, x2, kernel=kernel, last_dim_is_batch=last_dim_is_batch, **params)

    def _size(self) -> torch.Size:
        """
        Compute the size of the tensors handled by this class.
        """
        backup_x1 = self.x1.clone()
        backup_x2 = self.x2.clone()
        self.x1 = self.x1[..., 0]
        self.x2 = self.x2[..., 0]
        return_value = super()._size()
        self.x1 = backup_x1
        self.x2 = backup_x2
        return return_value


class HigherRankKernel(ScaledRBFKernel):
    """
    Define a kernel that applies to more than just vectors.
    """

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, last_dim_is_batch: bool = False, diag: bool = False, **params: Any
    ) -> torch.Tensor:
        """
        Evaluate the kernel given two tensors.

        :param x1: First tensor to evaluate
        :param x2: Second tensor to evaluate
        :param last_dim_is_batch: If true, the final dimension in the input tensors is considered the batch dimension
        :param diag: If true, we only compute the diagonal of the kernel matrix
        :return: Torch tensor holding kernel evaluation between inputs
        """
        return super().forward(
            x1.reshape(x1.shape[0], 4),
            x2.reshape(x2.shape[0], 4),
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
            **params,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Perform forward pass of the kernel.
        """
        return_tensor = super().__call__(*args, **kwargs)
        if isinstance(return_tensor, LazyEvaluatedKernelTensor):
            return_tensor = TwoDimensionalLazyEvaluatedKernelTensor.from_lazy_evaluated_kernel_tensor(return_tensor)
        return return_tensor


class HigherRankMean(ConstantMean):
    """
    Define a high rank mean class.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """
        Perform a forward pass of the mean class.

        :param input: Tensor we wish to evaluate the mean class at
        :return: Mean value given input tensor
        """
        return super().forward(input.reshape(input.shape[0], 4))


@HigherRankFeatures(2)
@DisableStandardScaling(ignore_all=True)
class Rank2Controller(GaussianGPController):
    pass


class BasicTests(unittest.TestCase):
    """
    Basic tests for the HigherRankFeatures decorator.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Define data shared across tests."""
        rng = get_default_rng()
        cls.dataset = HigherRankSyntheticDataset(rng=rng)

        cls.controller = Rank2Controller(
            cls.dataset.train_x,
            cls.dataset.train_y,
            HigherRankKernel,
            cls.dataset.train_y_std,
            mean_class=HigherRankMean,
            rng=rng,
        )

        cls.train_y_mean = cls.dataset.train_y.mean()
        cls.train_y_std = cls.dataset.train_y.std()

        cls.controller.fit(10)

    def test_posterior_shape(self) -> None:
        """Test that the posterior shape is as expected when using higher rank features."""
        posterior = self.controller.posterior_over_point(self.dataset.test_x)
        mean, _, _ = posterior.confidence_interval()
        self.assertEqual(mean.shape, self.dataset.test_y.shape)
