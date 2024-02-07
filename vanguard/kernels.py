"""
Vanguard includes class:`gpytorch.kernels.Kernel` subclasses which are recommended for use in controllers.
"""
from gpytorch import constraints, kernels
import torch


class ScaledRBFKernel(kernels.ScaleKernel):
    """
    The recommended starting place for a kernel.
    """
    def __init__(self, batch_shape=torch.Size(), ard_num_dims=None):
        """
        Initialise self.

        :param torch.Size,tuple batch_shape: The batch shape. Defaults to no batching.
        :param int,None ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
                                        Defaults to none.
        """
        super().__init__(kernels.RBFKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape), batch_shape=batch_shape)


class PeriodicRBFKernel(kernels.ScaleKernel):
    """
    An RBF kernel with a periodic element.
    """
    def __init__(self):
        """Initialise self."""
        super().__init__(kernels.RBFKernel() + kernels.ScaleKernel(kernels.RBFKernel() * kernels.PeriodicKernel()))


class TimeSeriesKernel(kernels.AdditiveKernel):
    """
    A kernel suited to time series.
    """
    def __init__(self, time_dimension=0):
        """
        Initialise self.
        """
        scaled_rbf_t = kernels.ScaleKernel(kernels.RBFKernel(active_dims=[time_dimension]))
        scaled_periodic_rbf = kernels.ScaleKernel(
            kernels.PeriodicKernel(active_dims=[time_dimension])
            * kernels.RBFKernel(active_dims=[time_dimension]))
        scaled_constrained_rbf = kernels.ScaleKernel(
            kernels.RBFKernel(active_dims=[time_dimension]), lengthscale_constraint=constraints.Interval(1, 14))
        scaled_linear_t = kernels.ScaleKernel(kernels.LinearKernel(active_dims=[time_dimension]))
        kernel_t = scaled_rbf_t + scaled_periodic_rbf + scaled_constrained_rbf

        super().__init__(scaled_linear_t, kernel_t)
