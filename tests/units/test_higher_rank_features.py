"""
Tests for the HigherRankFeatures decorator.
"""
import unittest

from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.means import ConstantMean

from vanguard.datasets.synthetic import HigherRankSyntheticDataset
from vanguard.features import HigherRankFeatures
from vanguard.kernels import ScaledRBFKernel
from vanguard.standardise import DisableStandardScaling
from vanguard.vanilla import GaussianGPController


class TwoDimensionalLazyEvaluatedKernelTensor(LazyEvaluatedKernelTensor):

    @classmethod
    def from_lazy_evaluated_kernel_tensor(cls, lazy_tensor):
        kernel = lazy_tensor.kernel
        x1 = lazy_tensor.x1
        x2 = lazy_tensor.x2
        last_dim_is_batch = lazy_tensor.last_dim_is_batch
        params = lazy_tensor.params
        return cls(x1, x2, kernel=kernel, last_dim_is_batch=last_dim_is_batch, **params)

    def _size(self):
        backup_x1 = self.x1.clone()
        backup_x2 = self.x2.clone()
        self.x1 = self.x1[..., 0]
        self.x2 = self.x2[..., 0]
        return_value = super()._size()
        self.x1 = backup_x1
        self.x2 = backup_x2
        return return_value


class HigherRankKernel(ScaledRBFKernel):
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        return super().forward(x1.reshape(x1.shape[0], 4), x2.reshape(x2.shape[0], 4), diag=diag,
                               last_dim_is_batch=last_dim_is_batch, **params)

    def __call__(self, *args, **kwargs):
        return_tensor = super().__call__(*args, **kwargs)
        if isinstance(return_tensor, LazyEvaluatedKernelTensor):
            return_tensor = TwoDimensionalLazyEvaluatedKernelTensor.from_lazy_evaluated_kernel_tensor(
                return_tensor)
        return return_tensor


class HigherRankMean(ConstantMean):
    def forward(self, x, *args, **kwargs):
        return super().forward(x.reshape(x.shape[0], 4), *args, **kwargs)


@HigherRankFeatures(2)
@DisableStandardScaling(ignore_all=True)
class Rank2Controller(GaussianGPController):
    pass


class BasicTests(unittest.TestCase):
    """
    Basic tests for the HigherRankFeatures decorator.
    """
    @classmethod
    def setUpClass(cls):
        """Code to run before all tests."""
        cls.dataset = HigherRankSyntheticDataset()

        cls.controller = Rank2Controller(cls.dataset.train_x, cls.dataset.train_y,
                                         HigherRankKernel, cls.dataset.train_y_std,
                                         mean_class=HigherRankMean)

        cls.train_y_mean = cls.dataset.train_y.mean()
        cls.train_y_std = cls.dataset.train_y.std()

        cls.controller.fit(10)

    def test_posterior_shape(self):
        posterior = self.controller.posterior_over_point(self.dataset.test_x)
        mean, lower, upper = posterior.confidence_interval()
        self.assertEqual(mean.shape, self.dataset.test_y.shape)
