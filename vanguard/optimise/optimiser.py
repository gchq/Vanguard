"""
Contains the SmartOptimiser class.
"""
from collections import deque
from functools import total_ordering
from heapq import heappush, heappushpop, nlargest
import inspect
from typing import TypeVar, Generic, Type, Optional, Generator, Any, Callable, Deque, List, Dict

import numpy as np
import torch.nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

OptimiserT = TypeVar("OptimiserT", bound=Optimizer)
class SmartOptimiser(Generic[OptimiserT]):
    """
    A smart wrapper around the standard optimisers found in PyTorch which can enable early stopping.

    .. warning::
        When setting the learning rate, using the :meth:`learning_rate` property,
        the parameters for each registered module are re-initialised.
    """

    _stored_initial_state_dicts: Dict[Module, Dict[str, Tensor]]
    last_n_losses: Deque[float]

    def __init__(self,
                 optimiser_class: Type[OptimiserT],
                 *initial_modules: Module,
                 early_stop_patience: Optional[int] = None,
                 **optimiser_kwargs
                 ):
        """
        Initialise self.

        :param type optimiser_class: An uninstantiated subclass of class:`torch.optim.Optimizer` to be used
            to create the internal optimiser.
        :param torch.nn.Module initial_modules: Initial modules whose parameters will be added to the
            internal optimiser.
        :param int,None early_stop_patience: How many consecutive gradient steps of worsening loss to allow before
            stopping early. Defaults to ``None`` which disables early stopping.
        :param optimiser_kwargs: Additional keyword arguments to be passed to the internal optimiser.
        """
        self._internal_optimiser_class = optimiser_class
        self._internal_optimiser_kwargs = optimiser_kwargs
        self._learning_rate = self._internal_optimiser_kwargs.pop("lr", 0.1)
        self._early_stop_patience = early_stop_patience

        self._stored_initial_state_dicts = {}
        self.last_n_losses = self._get_last_n_losses_structure(self._early_stop_patience)

        initial_parameters = []
        for module in initial_modules:
            self._cache_module_parameters(module)
            initial_parameters.append({"params": module.parameters()})
        self._internal_optimiser = self._internal_optimiser_class(initial_parameters, lr=self._learning_rate,
                                                                  **self._internal_optimiser_kwargs)

        self._set_step_method()

    @property
    def learning_rate(self) -> float:
        """Return the learning rate."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Set the value of the learning rate."""
        self._learning_rate = value
        self.reset()

    def parameters(self) -> Generator[Any, None, None]:
        """Get all parameters known to the optimiser."""
        for param_group in self._internal_optimiser.param_groups:
            yield from param_group["params"]

    def reset(self) -> None:
        """Reset everything."""
        self._reset_module_parameters()
        self._reset_internal_optimiser()
        self.last_n_losses = self._get_last_n_losses_structure(self._early_stop_patience)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Set the gradients of all optimized class:`torch.Tensor`s to zero."""
        return self._internal_optimiser.zero_grad(set_to_none=set_to_none)

    def step(self, loss: float, closure: Optional[Callable[[], float]] = None):
        """Perform a single optimisation step."""
        step_result = self._step(loss, closure=closure)
        self.last_n_losses.append(float(loss))

        no_improvement = self.last_n_losses[0] <= min(self.last_n_losses)
        if no_improvement:
            print_friendly_losses = ", ".join(f"{loss:.3f}" for loss in self.last_n_losses)
            raise NoImprovementError(f"Stopping early due to no improvement on {len(self.last_n_losses) - 1} "
                                     f"consecutive steps: [{print_friendly_losses}]")
        return step_result

    def register_module(self, module: Module):
        """Register the parameters for a module."""
        self._cache_module_parameters(module)
        parameters = {"params": module.parameters()}
        self._internal_optimiser.add_param_group(parameters)

    def update_registered_module(self, module: Module):
        """Update the parameters of a registered module if the module has been modified."""
        if module not in self._stored_initial_state_dicts:
            raise KeyError("Trying to update a module that isn't registered. Use register_module instead.")
        self._cache_module_parameters(module)
        self._reset_internal_optimiser()

    def set_parameters(self):
        """Tidy up after optimisation is completed."""
        pass

    def _reset_module_parameters(self) -> None:
        """
        Load afresh the stored initialisation values for all registered modules' parameters into the module.

        .. note::
            Calling this in isolation will restore the initialised values for all parameters, but it
            does not reset the optimiser. To do this, call :meth:`_reset_internal_optimiser` additionally.
        """
        for module, state_dict in self._stored_initial_state_dicts.items():
            module.load_state_dict(state_dict)

    def _reset_internal_optimiser(self) -> None:
        """
        Reset the internal optimiser.

        .. note::
            Calling this in isolation will not affect the current value of the parameters as learned
            thus far. To reset these, call :meth:`_reset_module_parameters` additionally.
        """
        parameters = [{"params": module.parameters()} for module in self._stored_initial_state_dicts]
        self._internal_optimiser = self._internal_optimiser_class(parameters, lr=self._learning_rate,
                                                                  **self._internal_optimiser_kwargs)

    def _step(self, loss: float, closure: Optional[Callable[[], float]] = None):
        """Perform a single optimisation step."""
        raise NotImplementedError

    def _cache_module_parameters(self, module: Module):
        """Cache the parameters for a module."""
        state_dict = module.state_dict()
        for parameter_name, parameter in state_dict.items():
            state_dict[parameter_name] = parameter.detach().clone()
        self._stored_initial_state_dicts[module] = state_dict

    def _set_step_method(self) -> None:
        """Create and set the :meth:`_step` method according to the internal optimiser."""
        internal_step_signature = inspect.signature(self._internal_optimiser.step)
        if "loss" in internal_step_signature.parameters:
            def new_step(loss, closure=None):
                """Pass the loss to the step function."""
                return self._internal_optimiser.step(loss, closure=closure)
        else:
            def new_step(_, closure=None):
                """Don't pass the loss to the step function."""
                return self._internal_optimiser.step(closure=closure)

        self._step = new_step

    @staticmethod
    def _get_last_n_losses_structure(n: Optional[int]) -> Deque[float]:
        """
        Get the structure which will contain the last :math`n` losses.

        Returns an instance of class:`collections.deque`.  This is
        always initialised with at least one ``nan`` value.  Whilst
        ``nan`` values occur in the structure, the minimum value will also
        be ``nan`` meaning that the minimum value will not be equal to the
        first element (because ``nan <= nan`` is ALWAYS ``False``.

        If ``n`` is a finite integer then these ``nan`` values will be
        popped from the structure after ``n+1`` additions. If ``n`` is
        ``None`` then the structure is infinite and this will never happen.

        :Example:
            >>> x = SmartOptimiser._get_last_n_losses_structure(2)
            >>> x
            deque([nan, nan, nan], maxlen=3)
            >>> for loss in range(2):
            ...     x.append(loss)
            ...     print(x[0], min(x), bool(x[0] <= min(x)))
            nan nan False
            nan nan False
            >>> x
            deque([nan, 0, 1], maxlen=3)
            >>> x.append(2)
            >>> print(x[0], min(x), bool(x[0] <= min(x)))
            0 0 True
            >>>
            >>> y = SmartOptimiser._get_last_n_losses_structure(None)
            >>> y
            deque([nan])
            >>> for loss in range(100):
            ...     x.append(loss)
            >>> print(y[0], min(y), bool(y[0] <= min(y)))
            nan nan False
        """
        if n is None:
            last_n_losses = deque([float("nan")], maxlen=None)
        else:
            max_length = n + 1
            last_n_losses = deque([float("nan")] * max_length, maxlen=max_length)

        return last_n_losses


@total_ordering
class Parameters:
    """
    Wrapped for module state_dicts and an objective value of their quality.
    """
    def __init__(self, module_state_dicts: Dict[Module, Dict[str, Tensor]], value: float = np.inf):
        """Initialise self."""
        self.parameters = {module: self._clone_state_dict(state_dict)
                           for module, state_dict in module_state_dicts.items()}
        self.priority_value = value

    def __lt__(self, other: "Parameters"):
        # TODO: this should check whether `isinstance(other, Parameters)` and if not return `NotImplemented`
        return self.priority_value < other.priority_value

    def __eq__(self, other: "Parameters"):
        # TODO: as for __lt__ above
        return self.priority_value == other.priority_value

    @staticmethod
    def _clone_state_dict(state_dict: Dict[str, Tensor]):
        """Detach and clone a state_dict so its tensors are not changed external to this class."""
        return {key: value.detach().clone() for key, value in state_dict.items()}


T = TypeVar("T")
class MaxLengthHeapQ(Generic[T]):
    """A heapq of fixed maximum length."""
    def __init__(self, max_length: int):
        """Initialise self."""
        self.max_length = max_length
        self.heap = []

    def push(self, item: T) -> None:
        """Push to the heapq."""
        if len(self.heap) < self.max_length:
            heappush(self.heap, item)
        else:
            heappushpop(self.heap, item)

    def nlargest(self, n: int) -> List[T]:
        """Get the top elements on the heapq."""
        return nlargest(n, self.heap)

    def best(self) -> T:
        """Get the top element."""
        return self.nlargest(1)[0]


class GreedySmartOptimiser(SmartOptimiser[OptimiserT], Generic[OptimiserT]):
    """
    Always choose parameters with the minimum loss value, regardless of the iteration at which they occur.

    .. note::
        This is the default smart optimiser for some class:`vanguard.vanilla.GaussianGPController.
        To disable the greedy loss behaviour and revert to keeping the parameters at the final iteration
        of training, using class:`vanguard.optimise.optimiser.SmartOptimiser` or a different subclass
        thereof.

    """
    N_RETAINED_PARAMETERS = 1

    def __init__(self,
                 optimiser_class: Type[OptimiserT],
                 *initial_modules: Module,
                 early_stop_patience: Optional[int] = None,
                 **optimiser_kwargs
                 ):
        super().__init__(
            optimiser_class,
            *initial_modules,
            early_stop_patience=early_stop_patience,
            **optimiser_kwargs
        )
        self._top_n_parameters = MaxLengthHeapQ(self.N_RETAINED_PARAMETERS)

    def step(self, loss: float, closure: Optional[Callable[[], float]] = None):
        """Step the optimiser and update the record best parameters."""
        super().step(loss, closure=closure)
        state_dicts = {module: module.state_dict() for module in self._stored_initial_state_dicts}
        loss_at_current_step = self.last_n_losses[-1]
        parameters = Parameters(state_dicts, -loss_at_current_step)
        self._top_n_parameters.push(parameters)

    def set_parameters(self) -> None:
        """Tidy up after optimisation by setting the parameters to the best."""
        best_parameters = self._top_n_parameters.best().parameters
        for module, state_dict in best_parameters.items():
            module.load_state_dict(state_dict)

    def reset(self) -> None:
        """Reset all parameters to start values and clear record of best parameters."""
        super().reset()
        self._top_n_parameters = MaxLengthHeapQ(self.N_RETAINED_PARAMETERS)


class NoImprovementError(RuntimeError):
    """Raised when the loss of the model is consistently increasing."""
    pass
