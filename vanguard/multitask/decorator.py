# © Crown Copyright GCHQ
#
# Licensed under the GNU General Public License, version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Enabling multitask Gaussian processes.

The :class:`~vanguard.multitask.decorator.Multitask` decorator
converts a controller class into a multitask controller.
"""

from typing import Any, Dict, Optional, Type, TypeVar

import torch
from gpytorch.kernels import Kernel, MultitaskKernel
from gpytorch.means import ConstantMean, Mean, MultitaskMean
from torch import Tensor

from vanguard import utils
from vanguard.base import GPController
from vanguard.decoratorutils import Decorator, process_args, wraps_class
from vanguard.multitask.kernel import BatchCompatibleMultitaskKernel
from vanguard.multitask.models import (
    independent_variational_multitask_model,
    lmc_variational_multitask_model,
    multitask_model,
)
from vanguard.variational import VariationalInference

ControllerT = TypeVar("ControllerT", bound=GPController)
T = TypeVar("T")


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

    def __init__(self, num_tasks: int, lmc_dimension: Optional[int] = None, rank: int = 1, **kwargs: Any) -> None:
        """
        Initialise self.

        :param num_tasks: The number of tasks (i.e. y-value dimension).
        :param lmc_dimension: If using LMC (linear model of co-regionalisation), how many latent dimensions
            to use. Bigger means a more complicated model. Should probably be at least as big as the number of
            tasks, unless you want to specifically make low-rank assumptions about the relationship between tasks.
            Default (None) means LMC is not used at all.
        :param rank: The rank of the task-task covar matrix in a Kronecker product multitask kernel. Only relevant
            for exact GP inference.
        """
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)
        self.num_tasks = num_tasks
        self.lmc_dimension = lmc_dimension
        self.rank = rank

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
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

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                all_parameters_as_kwargs = process_args(super().__init__, *args, **kwargs)
                self.rng = utils.optional_random_generator(all_parameters_as_kwargs.pop("rng", None))

                # it's OK to access self.gp_model_class as it's set in super().__init__ above
                original_gp_model_class = self.gp_model_class  # pylint: disable=access-member-before-definition
                if is_variational:
                    if decorator.lmc_dimension is not None:
                        gp_model_class = lmc_variational_multitask_model(original_gp_model_class)
                    else:
                        gp_model_class = independent_variational_multitask_model(original_gp_model_class)
                else:
                    gp_model_class = original_gp_model_class

                # Pyright cannot resolve dynamic base class
                @multitask_model
                class MultitaskGPModelClass(gp_model_class):  # pyright: ignore[reportGeneralTypeIssues]
                    """Multitask version of gp_model_class."""

                self.gp_model_class = MultitaskGPModelClass

                self.num_tasks = decorator.num_tasks
                mean_class = all_parameters_as_kwargs.pop("mean_class", ConstantMean)
                kernel_class = all_parameters_as_kwargs.pop("kernel_class")

                kernel_kwargs = all_parameters_as_kwargs.get("kernel_kwargs", {})
                mean_kwargs = all_parameters_as_kwargs.get("mean_kwargs", {})

                if is_variational:
                    kernel_class = _batchify(kernel_class, kernel_kwargs, decorator.num_tasks, decorator.lmc_dimension)
                    mean_class = _batchify(mean_class, mean_kwargs, decorator.num_tasks, decorator.lmc_dimension)
                else:
                    kernel_class = _multitaskify_kernel(kernel_class, decorator.num_tasks, decorator.rank)

                try:
                    mean_class = self._match_mean_shape_to_kernel(mean_class, kernel_class, mean_kwargs, kernel_kwargs)
                except TypeError as exc:
                    # Check for batch shape mismatches and reraise with a more informative message.
                    if "batch_shape" in mean_kwargs:
                        batch_shape = mean_kwargs["batch_shape"]
                        if not isinstance(batch_shape, torch.Size):
                            msg = (
                                f"Expected mean_kwargs['batch_shape'] to be of type `torch.Size`; "
                                f"got `{batch_shape.__class__.__name__}` instead"
                            )
                            raise TypeError(msg) from exc
                    # If it's some other TypeError, just re-raise it.
                    raise
                likelihood_kwargs = all_parameters_as_kwargs.pop("likelihood_kwargs", {})
                likelihood_kwargs["num_tasks"] = decorator.num_tasks
                gp_kwargs = all_parameters_as_kwargs.pop("gp_kwargs", {})
                gp_kwargs["num_tasks"] = decorator.num_tasks

                super().__init__(
                    kernel_class=kernel_class,
                    mean_class=mean_class,
                    likelihood_kwargs=likelihood_kwargs,
                    gp_kwargs=gp_kwargs,
                    rng=self.rng,
                    **all_parameters_as_kwargs,
                )

            @property
            def likelihood_noise(self) -> Tensor:
                """Return the fixed noise of the likelihood."""
                try:
                    return self._likelihood.fixed_noise
                except AttributeError as exc:
                    raise AttributeError(
                        "'fixed_noise' appears to have not been set yet. This can be set "
                        "with the `likelihood_noise` method"
                    ) from exc

            @likelihood_noise.setter
            def likelihood_noise(self, value: Tensor) -> None:
                """Set the fixed noise of the likelihood."""
                self._likelihood.fixed_noise = value

            @staticmethod
            def _match_mean_shape_to_kernel(
                mean_class: Type[Mean],
                kernel_class: Type[Kernel],
                mean_kwargs: Dict[str, Any],
                kernel_kwargs: Dict[str, Any],
            ) -> Type[Mean]:
                """
                Construct a mean class suitable for multitask GPs that matches the form of the kernel, if possible.

                :param mean_class: An uninstantiated :class:`gpytorch.means.Mean`.
                :param kernel_class: An uninstantiated :class:`gpytorch.kernels.Kernel`.
                :param mean_kwargs: Keyword arguments to be passed to the mean_class constructor.
                :param kernel_kwargs: Keyword arguments to be passed to the kernel_class constructor.
                :returns: An uninstantiated :class:`gpytorch.means.Mean` like mean_class but modified to have the
                    same form/shape as kernel_class, if possible.
                :raises TypeError: If the supplied mean_class has a batch_shape and it doesn't match the batch_shape of
                    the kernel_class, or is a :class:`gpytorch.kernels.MultitaskKernel` and has num_tasks which doesn't
                    match that of the kernel_class.
                """
                example_kernel = kernel_class(**kernel_kwargs)
                example_mean = mean_class(**mean_kwargs)

                if isinstance(example_kernel, MultitaskKernel):
                    return _multitaskify_mean(mean_class, decorator.num_tasks)
                if len(example_kernel.batch_shape) > 0 and example_mean.batch_shape != example_kernel.batch_shape:
                    msg = (
                        f"The provided mean has batch_shape {example_mean.batch_shape} but the "
                        f"provided kernel has batch_shape {example_kernel.batch_shape}. "
                        f"They must match."
                    )
                    raise ValueError(msg)
                return mean_class

        # Pyright does not detect that wraps_class renames InnerClass
        return InnerClass  # pyright: ignore [reportReturnType]


def _batchify(module_class: Type[T], _kwargs: Dict[str, Any], num_tasks: int, lmc_dimension: Optional[int]) -> Type[T]:
    """
    Add a batch shape to a class so it can be used for multitask variational GPs.

    :param module_class: The class to batchify, typically a kernel or mean.
    :param _kwargs: Remaining in signature for compatibility.
    :param num_tasks: The number of tasks for the multitask GP.
    :param lmc_dimension: The number of LMC dimensions (if using LMC).

    :returns: The adapted class.
    """
    batch_size = lmc_dimension if lmc_dimension is not None else num_tasks

    @wraps_class(module_class)
    class InnerClass(module_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            batch_shape = kwargs.pop("batch_shape", torch.Size([])) + torch.Size([batch_size])
            kwargs["batch_shape"] = batch_shape
            super().__init__(*args, **kwargs)

    # Pyright does not detect that wraps_class renames InnerClass
    return InnerClass  # pyright: ignore [reportReturnType]


def _multitaskify_kernel(kernel_class: Type[Kernel], num_tasks: int, rank: int = 1) -> Type[MultitaskKernel]:
    """
    If necessary, make a kernel multitask using the GPyTorch Multitask kernel.

    :param kernel_class: The kernel to multitaskify.
    :param num_tasks: The number of tasks for the multitask GP.
    :param rank: The rank of the task-task covariance matrix.

    :returns: The adapted kernel class.
    """
    if issubclass(kernel_class, MultitaskKernel):
        return kernel_class
    else:
        rank = min(num_tasks, rank)

        class InnerKernelClass(BatchCompatibleMultitaskKernel):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(kernel_class(*args, **kwargs), num_tasks=num_tasks, rank=rank, **kwargs)

        return InnerKernelClass


def _multitaskify_mean(mean_class: Type[Mean], num_tasks: int) -> Type[MultitaskMean]:
    """
    If necessary, make a mean multitask using the GPyTorch Multitask mean.

    :param mean_class: The mean to multitaskify.
    :param num_tasks: The number of tasks for the multitask GP.

    :returns: The adapted mean class.
    """
    if issubclass(mean_class, MultitaskMean):
        return mean_class
    else:

        class InnerMeanClass(MultitaskMean):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(mean_class(*args, **kwargs), num_tasks=num_tasks)

        return InnerMeanClass
