"""
Contains decorators for torch optimisers to apply LR schedulers as part of the optimisation step.
"""
import inspect


class ApplyLearningRateScheduler:
    """
    Apply a torch learning rate scheduler to a torch optimiser.

    The scheduler is stepped at each step of optimiser.
    """
    def __init__(self, scheduler_class, *args, **kwargs):
        """
        :param type scheduler_class: The (uninstantiated) torch learning rate scheduler to be used.
        """
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = kwargs
        self.scheduler_args = args
        self.scheduler_takes_loss = "metrics" in inspect.signature(scheduler_class.step).parameters

    def __call__(self, cls):
        """Apply scheduler to optimiser."""
        scheduler_class = self.scheduler_class
        scheduler_kwargs = self.scheduler_kwargs
        scheduler_args = self.scheduler_args
        scheduler_step_func = self._step_scheduler_with_loss if self.scheduler_takes_loss else self._step_scheduler

        class InnerClass(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._applied_scheduler = scheduler_class(self, *scheduler_args, **scheduler_kwargs)

            def step(self, loss, closure=None):
                ret = super().step(closure=closure)
                scheduler_step_func(self._applied_scheduler, loss)
                return ret
        return InnerClass

    @staticmethod
    def _step_scheduler(scheduler, loss):
        scheduler.step()

    @staticmethod
    def _step_scheduler_with_loss(scheduler, loss):
        scheduler.step(loss)
