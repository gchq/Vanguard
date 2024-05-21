"""
All warp functions should subclass this :class:`WarpFunction` class.
"""

import copy
from functools import wraps
from itertools import chain
from typing import Callable, Iterator, List, TypeVar, Union

import gpytorch
import torch
from typing_extensions import Never, Self


class WarpFunction(gpytorch.Module):
    """
    Base module for warp functions.

    Subclasses must implement the :meth:`forward` and :meth:`inverse` methods. Optionally, the
    :meth:`deriv` method can be implemented. If not, then it defaults to autograd, which is significantly slower.

    :Example:
        >>> # New warp functions should inherit from the WarpFunction class:
        >>> class AddTwo(WarpFunction):
        ...     def forward(self, y):
        ...         return y + 2
        ...
        ...     def inverse(self, x):
        ...          return x - 2
        ...
        ...     def deriv(self, y):
        ...         return 1
        >>>
        >>> # Warp functions can be composed together:
        >>> add_four = AddTwo() @ AddTwo()
        >>> add_four.forward(torch.Tensor([0]))
        tensor([4.])
        >>>
        >>> # You can also compose copies of the same function:
        >>> add_ten = AddTwo() @ 5
        >>> add_ten.forward(torch.Tensor([0]))
        tensor([10.])
    """

    def __matmul__(self, other: Union["WarpFunction", int]) -> "WarpFunction":
        if isinstance(other, WarpFunction):
            return self.compose(other)
        elif isinstance(other, int):
            try:
                return self.compose_with_self(other)
            except ValueError:
                pass

        return NotImplemented

    @property
    def components(self) -> List["WarpFunction"]:
        """Get the components of the composition."""
        try:
            components = self.old_warp_left.components + self.old_warp_right.components
        except AttributeError:
            components = [self]
        return components

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Pass an input tensor through the warp function.

        :param y: An input tensor.
        :returns: A tensor of same shape as y.
        """
        raise NotImplementedError("Using base class Warp.")

    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        """
        Return the derivative of the warp function at a point, y.

        :param y: An input tensor.
        :returns: A tensor of same shape as y, the warp function's gradient at y.
        """
        g_y = y.detach().clone()
        g_y.requires_grad = True
        x = self.forward(g_y).sum()
        x.backward()
        assert g_y.grad is not None
        return g_y.grad

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the inverse of the warp function at a point, x.

        :param x: An input tensor.
        :returns: A tensor of same shape as x, the warp function's inverse at x.
        """
        raise NotImplementedError("Using base class Warp.")

    def compose_with_self(self, n: int) -> "WarpFunction":
        """
        Repeatedly compose a warp function with itself.

        :param n: The number of times to compose.
        :return: A new WarpFunction instance with composed functions.
        :raises ValueError: If ``n`` is negative.

        .. note::
            When ``n == 0``, this method will return the identity warp.

        .. warning::
            The warp functions are not copied before composition, meaning that
            each component of the returned warp function will be the same object.
            When applied to a controller class with the
            :class:`~vanguard.warps.SetWarp` decorator, the warp function
            (and its components) will be copied and this will no longer be an issue.
        """
        if n > 0:
            new_warp = self
            for _ in range(n - 1):
                new_warp = new_warp @ self
        elif n == 0:
            new_warp = _IdentityWarpFunction()
        else:
            raise ValueError("'n' cannot be negative.")
        return new_warp

    def compose(self, other: "WarpFunction") -> "WarpFunction":
        """
        Compose with another warp function.

        :param other: The other warp function.
        :return: A new WarpFunction instance with composed functions.

        .. note::
            For convenience, it is often easier to use the ``@`` operator in place of :meth:`compose`.

        :Example:
            >>> warp_1, warp_2 = WarpFunction(), WarpFunction()
            >>>
            >>> # this will be the equivalent of warp_1(warp_2(...))
            >>> composed_warp = warp_1 @ warp_2
        """
        new_warp = WarpFunction()
        new_warp.old_warp_left = self
        new_warp.old_warp_right = other

        try:
            new_warp.forward = _composition_factory(self, other)
            new_warp.inverse = _composition_factory(other.inverse, self.inverse)
            new_warp.deriv = _multiply_factory(_composition_factory(self.deriv, other), other.deriv)
            # Overwrite parameters method with an iterator
            new_warp.parameters = new_warp._combined_parameters  # pyright: ignore [reportAttributeAccessIssue]
        except AttributeError:
            if not isinstance(other, WarpFunction):
                raise TypeError("Must be passed a valid WarpFunction instance.")
            else:
                raise

        return new_warp

    def copy(self) -> Self:
        """Return a copy guaranteed to have distinct parameters."""
        try:
            return self.old_warp_left.copy() @ self.old_warp_right.copy()
        except AttributeError:
            return copy.deepcopy(self)

    def freeze(self) -> Self:
        """
        Return a copy of the warp with frozen parameters.

        .. note::
            We override :attr:`torch.nn.Module.parameters` to return an empty generator, so no amount of
            ``return_grad=True`` will make the parameters trainable again. This is the most reliable way of freezing
            parameters and keeping them frozen in downstream usage.

        :return: A copy of self with parameters frozen.
        """
        new_warp = self.copy()
        # Overwrite parameters method with an iterator
        new_warp.parameters = _empty_generator  # pyright: ignore [reportAttributeAccessIssue]
        return new_warp

    def _combined_parameters(self) -> Iterator[torch.nn.Module.parameters]:
        """
        Return the combined parameters of the composition.

        Used in composition warps to override the default :attr:`torch.nn.Module.parameters` so that
        frozen functions remain frozen under composition.
        """
        return chain(self.old_warp_left.parameters(), self.old_warp_right.parameters())


class _IdentityWarpFunction(WarpFunction):
    """
    The identity map as a warp.
    """

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(y)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MultitaskWarpFunction(WarpFunction):
    """
    Module for multitask warp functions.

    It is expected that the warps will be applied element-wise to vectors,
    with possibly a different warp for each dimension.

    1d warps (subclasses of py:class:`~vanguard.warps.warpfunctions.WarpFunction`) for each dimension are
    just passed to this module's constructor.

    :Example:
        >>> # New warp functions should inherit from the WarpFunction class:
        >>> class AddTwo(WarpFunction):
        ...     def forward(self, y):
        ...         return y + 2
        ...
        ...     def inverse(self, x):
        ...          return x - 2
        ...
        ...     def deriv(self, y):
        ...         return 1
        >>>
        >>> # Warp functions can be composed together:
        >>> add_two = AddTwo()
        >>> add_two.forward(torch.Tensor([0]))
        tensor([2.])
        >>> add_four = AddTwo() @ AddTwo()
        >>> add_four.forward(torch.Tensor([0]))
        tensor([4.])
        >>> multitask_warp = MultitaskWarpFunction(add_two, add_four)
        >>> multitask_warp.forward(torch.Tensor([[0, 1], [-2, -3]]))
        tensor([[2., 5.],
                [0., 1.]])
    """

    def __init__(self, *warps: WarpFunction):
        """
        Initialise self.

        :param warps: The warp functions to be applied.
        """
        super().__init__()
        self.warps = torch.nn.ModuleList(warps)

    @property
    def num_tasks(self) -> int:
        return len(self.warps)

    def forward(self, stack_of_y: torch.Tensor) -> torch.Tensor:
        """
        Pass an input tensor through the warp function.

        :param stack_of_y: A stack of input tensors.
        :returns: A stack of tensors in the same shape as stack_of_y.
        """
        return torch.stack([warp.forward(task_y).squeeze() for warp, task_y in zip(self.warps, stack_of_y.T)], -1)

    def deriv(self, y: torch.Tensor) -> torch.Tensor:
        """
        Return the derivative of the warp function at a point, y.

        :param y: An input tensor.
        :returns: A tensor of same shape as y, the warp function's gradient at y.
        """
        return torch.stack([warp.deriv(task_y).squeeze() for warp, task_y in zip(self.warps, y.T)], -1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the inverse of the warp function at a point, x.

        :param x: An input tensor.
        :returns: A tensor of same shape as x, the warp function's inverse at x.
        """
        return torch.stack([warp.inverse(task_x).squeeze() for warp, task_x in zip(self.warps, x.T)], -1)

    def compose(self, other: "WarpFunction") -> "MultitaskWarpFunction":
        """
        Compose with another warp function.

        :param other: The other warp function.
        :return: A new MultitaskWarpFunction instance with composed functions task-wise.

        .. note::
            For convenience, it is often easier to use the ``@`` operator in place of :meth:`compose`.

        :Example:
            >>> warp_1, warp_2 = MultitaskWarpFunction(), MultitaskWarpFunction()
            >>>
            >>> # this will be the equivalent of warp_1(warp_2(...)) task-wise.
            >>> composed_warp = warp_1 @ warp_2
        """
        try:
            new_task_warps = [warp.compose(other_warp) for warp, other_warp in zip(self.warps, other.warps)]
        except AttributeError:
            if not isinstance(other, MultitaskWarpFunction):
                raise TypeError("Must be passed a valid MultitaskWarpFunction instance.")
            elif not all(isinstance(warp, WarpFunction) for warp in other.warps):
                raise TypeError(
                    "All of the per-task warps for the passed MultitaskWarpFunction must be valid instances"
                    "of WarpFunction."
                )
            else:
                raise
        new_warp = MultitaskWarpFunction(*new_task_warps)
        return new_warp

    def compose_with_self(self, n: int) -> "MultitaskWarpFunction":
        """
        Repeatedly compose a warp function with itself.

        :param n: The number of times to compose.
        :return: A new WarpFunction instance with composed functions.
        :raises ValueError: If ``n`` is negative.

        .. note::
            When ``n == 0``, this method will return the identity warp.

        .. warning::
            The warp functions are not copied before composition, meaning that
            each component of the returned warp function will be the same object.
            When applied to a controller class with the
            :class:`~vanguard.warps.SetWarp` decorator, the warp function
            (and its components) will be copied and this will no longer be an issue.
        """
        if n > 0:
            new_warp = self
            for _ in range(n - 1):
                # Operator usage defined in __matmul__
                new_warp = new_warp @ self  # pyright: ignore [reportOperatorIssue]
        elif n == 0:
            new_warp = type(self)([_IdentityWarpFunction()] * self.num_tasks)
        else:
            raise ValueError("'n' cannot be negative.")
        return new_warp


ComposableT = TypeVar("ComposableT", WarpFunction, Callable)


def _composition_factory(f1: ComposableT, f2: ComposableT) -> ComposableT:
    """Return the function for f1(f2(x))."""

    @wraps(f1)
    def composition(*args):
        """Inner function."""
        return f1(f2(*args))

    return composition


def _multiply_factory(f1: ComposableT, f2: ComposableT) -> ComposableT:
    """Return the function for f1(x) * f2(x)."""

    @wraps(f1)
    def composition(*args):
        """Inner function."""
        return f1(*args) * f2(*args)  # pyright: ignore [reportOperatorIssue]

    return composition


def _empty_generator() -> Iterator[Never]:
    """Return an empty generator for convenience."""
    return iter(())
