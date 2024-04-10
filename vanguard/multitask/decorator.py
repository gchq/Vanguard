"""
Enabling multitask Gaussian processes.

The :class:`~vanguard.multitask.decorator.Multitask` decorator
converts a controller class into a multitask controller.
"""
import torch
from gpytorch.kernels import MultitaskKernel
from gpytorch.means import ConstantMean, MultitaskMean

from ..base import GPController
from ..decoratorutils import Decorator, process_args, wraps_class
from ..variational import VariationalInference
from .kernel import BatchCompatibleMultitaskKernel
from .models import independent_variational_multitask_model, lmc_variational_multitask_model, multitask_model


class Multitask(Decorator):
    """
    Make a GP multitask.

    :Example:
            >>> from vanguard.base import GPController
            >>>
            >>> @Multitask(num_tasks=2)
            ... class MyController(GPController):
            ...     pass
    """
    def __init__(self, num_tasks, lmc_dimension=None, rank=1, **kwargs):
        """
        Initialise self.

        :param int num_tasks: The number of tasks (i.e. y-value dimension).
        :param int,None lmc_dimension: If using LMC (linear model of co-regionalisation), how many latent dimensions
                                        to use. Bigger means a more complicated model. Should probably be at least
                                        as big as the number of tasks, unless you want to specifically make low-rank
                                        assumptions about the relationship between tasks.
                                        Default (None) means LMC isn't not used at all.
        :param int rank: The rank of the task-task covar matrix in a Kronecker product multitask kernel.
                            Only relevant for exact GP inference.
        """
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)
        self.num_tasks = num_tasks
        self.lmc_dimension = lmc_dimension
        self.rank = rank

    def _decorate_class(self, cls):
        decorator = self
        is_variational = VariationalInference in cls.__decorators__

        @wraps_class(cls)
        class InnerClass(cls):
            """
            A wrapper for converting a controller class to multitask.

            It is fairly lightweight, extracting the necessary information
            like number of tasks from the supplied data, converting means
            to multitask means and slightly modifying a few methods to deal
            with multitask Gaussian's etc.
            """
            def __init__(self, *args, **kwargs):
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                all_parameters_as_kwargs.pop("self")

                if is_variational:
                    if decorator.lmc_dimension is not None:
                        gp_model_class = lmc_variational_multitask_model(self.gp_model_class)
                    else:
                        gp_model_class = independent_variational_multitask_model(self.gp_model_class)
                else:
                    gp_model_class = self.gp_model_class

                @multitask_model
                class MultitaskGPModelClass(gp_model_class):
                    """Multitask version of gp_model_class."""
                    pass

                self.gp_model_class = MultitaskGPModelClass

                self.num_tasks = decorator.num_tasks
                mean_class = all_parameters_as_kwargs.pop("mean_class", ConstantMean)
                kernel_class = all_parameters_as_kwargs.pop("kernel_class")

                kernel_kwargs = all_parameters_as_kwargs.get("kernel_kwargs", {})
                mean_kwargs = all_parameters_as_kwargs.get("mean_kwargs", {})

                if is_variational:
                    kernel_class = _batchify(kernel_class, kernel_kwargs, decorator.num_tasks,
                                             decorator.lmc_dimension)
                    mean_class = _batchify(mean_class, mean_kwargs, decorator.num_tasks,
                                           decorator.lmc_dimension)
                else:
                    kernel_class = _multitaskify_kernel(kernel_class, decorator.num_tasks, decorator.rank)

                mean_class = self._match_mean_shape_to_kernel(mean_class, kernel_class, mean_kwargs, kernel_kwargs)

                likelihood_kwargs = all_parameters_as_kwargs.pop("likelihood_kwargs", {})
                likelihood_kwargs["num_tasks"] = decorator.num_tasks
                gp_kwargs = all_parameters_as_kwargs.pop("gp_kwargs", {})
                gp_kwargs["num_tasks"] = decorator.num_tasks

                super().__init__(kernel_class=kernel_class, mean_class=mean_class,
                                 likelihood_kwargs=likelihood_kwargs, gp_kwargs=gp_kwargs,
                                 **all_parameters_as_kwargs)

            @property
            def likelihood_noise(self):
                """Return the fixed noise of the likelihood."""
                return self._likelihood.fixed_noise

            @likelihood_noise.setter
            def likelihood_noise(self, value):
                """Set the fixed noise of the likelihood."""
                self._likelihood.fixed_noise = value

            @staticmethod
            def _match_mean_shape_to_kernel(mean_class, kernel_class, mean_kwargs, kernel_kwargs):
                """
                Construct a mean class suitable for multitask GPs that matches the form of the kernel, if possible.

                :param mean_class: An uninstantiated :class:`gpytorch.means.Mean`.
                :param kernel_class: An uninstantiated :class:`gpytorch.kernels.Kernel`.
                :param dict mean_kwargs: Keyword arguments to be passed to the mean_class constructor.
                :param dict kernel_kwargs: Keyword arguments to be passed to the kernel_class constructor.
                :returns: An uninstantiated class:`gpytorch.means.Mean` like mean_class but modified to have the
                          same form/shape as kernel_class, if possible.
                :rtype: type
                :raises TypeError: If the supplied mean_class has a batch_shape and it doesn't match the batch_shape of
                                    the kernel_class, or is a class:`gpytorch.kernels.MultitaskKernel` and has
                                    num_tasks which doesn't match that of the kernel_class.
                """
                example_kernel = kernel_class(**kernel_kwargs)
                example_mean = mean_class(**mean_kwargs)

                if isinstance(example_kernel, MultitaskKernel):
                    return _multitaskify_mean(mean_class, decorator.num_tasks)
                if len(example_kernel.batch_shape) > 0 and example_mean.batch_shape != example_kernel.batch_shape:
                    raise TypeError(f"The provided mean has batch_shape {example_mean.batch_shape} but the "
                                    f"provided kernel has batch_shape {example_kernel.batch_shape}. "
                                    f"They must match.")
                return mean_class
        return InnerClass


def _batchify(module_class, _kwargs, num_tasks, lmc_dimension):
    """
    Add a batch shape to a kernel so it can be used for multitask variational GPs.

    :param gpytorch.kernels.Kernel kernel_class: The kernel to batchify.
    :param dict _kwargs: Remaining in signature for compatibility.
    :param int num_tasks: The number of tasks for the multitask GP.
    :param int,None lmc_dimension: The number of LMC dimensions (if using LMC).

    :returns: The adapted kernel class.
    :rtype: gpytorch.kernels.Kernel
    """
    batch_size = lmc_dimension if lmc_dimension is not None else num_tasks

    class InnerClass(module_class):
        def __init__(self, *args, **kwargs):
            batch_shape = kwargs.pop("batch_shape", torch.Size([])) + torch.Size([batch_size])
            kwargs["batch_shape"] = batch_shape
            super().__init__(*args, **kwargs)
    return InnerClass


def _multitaskify_kernel(kernel_class, num_tasks, rank=1):
    """
    If necessary, make a kernel multitask using the GPyTorch Multitask kernel.

    :param gpytorch.kernels.Kernel kernel_class: The kernel to multitaskify.
    :param int num_tasks: The number of tasks for the multitask GP.
    :param int rank: The rank of the task-task covariance matrix.

    :returns: The adapted kernel class.
    :rtype: gpytorch.kernels.Kernel
    """
    if issubclass(kernel_class, MultitaskKernel):
        return kernel_class
    else:
        rank = min(num_tasks, rank)

        class InnerKernelClass(BatchCompatibleMultitaskKernel):
            def __init__(self, *args, **kwargs):
                super().__init__(kernel_class(*args, **kwargs), num_tasks=num_tasks, rank=rank, **kwargs)
        return InnerKernelClass


def _multitaskify_mean(mean_class, num_tasks):
    """
    If necessary, make a mean multitask using the GPyTorch Multitask mean.

    :param gpytorch.means.Mean mean_class: The mean to multitaskify.
    :param int num_tasks: The number of tasks for the multitask GP.

    :returns: The adapted mean class.
    :rtype: gpytorch.means.Means
    """
    if issubclass(mean_class, MultitaskMean):
        return mean_class
    else:

        class InnerMeanClass(MultitaskMean):
            def __init__(self, *args, **kwargs):
                super().__init__(mean_class(*args, **kwargs), num_tasks=num_tasks)
        return InnerMeanClass
