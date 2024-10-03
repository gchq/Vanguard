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
Contains decorators for torch optimisers to apply LR schedulers as part of the optimisation step.
"""

import inspect
from typing import Any, Callable, Generic, Optional, Type, TypeVar, Union, overload

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

OptimiserT = TypeVar("OptimiserT", bound=Optimizer)
LRSchedulerT = TypeVar("LRSchedulerT", bound=LRScheduler)


class ApplyLearningRateScheduler(Generic[LRSchedulerT]):
    """
    Apply a torch learning rate scheduler to a torch optimiser.

    The scheduler is stepped at each step of optimiser.
    """

    def __init__(self, scheduler_class: Type[LRSchedulerT], *args: Any, **kwargs: Any) -> None:
        """
        :param scheduler_class: The (uninstantiated) torch learning rate scheduler to be used.
        """
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = kwargs
        self.scheduler_args = args
        self.scheduler_takes_loss = "metrics" in inspect.signature(scheduler_class.step).parameters

    def __call__(self, cls: Type[OptimiserT]) -> Type[OptimiserT]:
        """Apply scheduler to optimiser."""
        scheduler_class = self.scheduler_class
        scheduler_kwargs = self.scheduler_kwargs
        scheduler_args = self.scheduler_args
        scheduler_step_func = self._step_scheduler_with_loss if self.scheduler_takes_loss else self._step_scheduler

        # Can't use @wraps_class here as it causes a unit test failure?
        class InnerClass(cls):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._applied_scheduler = scheduler_class(self, *scheduler_args, **scheduler_kwargs)

            @overload
            def step(self, loss: Union[float, torch.Tensor], closure: None) -> None: ...  # pragma: no cover

            @overload
            def step(
                self, loss: Union[float, torch.Tensor], closure: Callable[[], float]
            ) -> Union[float, torch.Tensor]: ...  # pragma: no cover

            def step(
                self, loss: Union[float, torch.Tensor], closure: Optional[Callable[[], float]] = None
            ) -> Optional[Union[float, torch.Tensor]]:
                ret = super().step(closure=closure)
                scheduler_step_func(self._applied_scheduler, loss)
                return ret

        return InnerClass

    @staticmethod
    def _step_scheduler(scheduler: LRSchedulerT, _) -> None:
        scheduler.step()

    @staticmethod
    def _step_scheduler_with_loss(scheduler: LRSchedulerT, loss: Union[float, torch.Tensor]) -> None:
        scheduler.step(loss)
