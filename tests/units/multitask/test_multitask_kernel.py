"""
Tests for the Multitask kernel class.
"""

import unittest

import gpytorch
import torch
from linear_operator.operators import KroneckerProductLinearOperator

from vanguard.kernels import ScaledRBFKernel
from vanguard.multitask.decorator import _multitaskify_kernel
from vanguard.multitask.kernel import BatchCompatibleMultitaskKernel


class KernelTests(unittest.TestCase):
    """
    Test forward calls with the BatchCompatibleMultitaskKernel class.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.num_tasks = 2
        self.kernel = BatchCompatibleMultitaskKernel(
            data_covar_module=ScaledRBFKernel(),
            num_tasks=self.num_tasks,
        )

    def test_last_batch_dim(self) -> None:
        """Test how BatchCompatibleMultitaskKernel handles the last dimension being the batch dimension."""
        with self.assertRaisesRegex(RuntimeError, "MultitaskKernel does not accept the last_dim_is_batch argument."):
            self.kernel(x1=torch.zeros([3, 2, 4]), x2=torch.ones([3, 2, 4]), last_dim_is_batch=True).to_dense()

    def test_batched_data(self) -> None:
        """Test how batched data is handled."""
        x = torch.ones([5, 4, 3, 2])
        with gpytorch.settings.lazily_evaluate_kernels(False):
            result = self.kernel(x1=x, x2=x)
        self.assertIsInstance(result, KroneckerProductLinearOperator)

    def test_kernel_conversion_null(self) -> None:
        """Test conversion of kernels to multitask kernels."""
        # If we try to convert an already multitask kernel, we should just get out what we put in
        self.assertIsInstance(
            _multitaskify_kernel(BatchCompatibleMultitaskKernel, self.num_tasks)(ScaledRBFKernel(), self.num_tasks),
            BatchCompatibleMultitaskKernel,
        )

        # If we try to convert a non-multitask kernel, we should not get out what we put in. Note that we start with a
        # scaled RBF kernel, but when we initialise the output of _multitaskify_kernel, it's now of a different type
        self.assertIsInstance(_multitaskify_kernel(ScaledRBFKernel, self.num_tasks)(), BatchCompatibleMultitaskKernel)
