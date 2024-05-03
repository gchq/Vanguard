"""
Contains the BaseDecorator class.
"""

import warnings
from inspect import getmembers, isfunction
from typing import Iterable, Set, Type, TypeVar

from . import errors

T = TypeVar("T")
DecoratorType = TypeVar("DecoratorType", bound="Decorator")


class Decorator:
    """
    A base class for a vanguard decorator.

    .. note::
        Decorating :class:`~vanguard.base.gpcontroller.GPController` classes is an
        extremely practical means of extending functionality.  However, many
        decorators are designed to work with a specific 'framework class', and
        any methods which have been added (or modified) to the decorated class
        can cause issues which may not be picked up at runtime.

        To mitigate this, any unexpected or modified methods (along with any
        other potential problems that the creator may wish to avoid) will emit
        a :class:`~vanguard.decoratorutils.errors.DecoratorWarning` or raise a
        :class:`~vanguard.decoratorutils.errors.DecoratorError`
        at runtime if the decorator calls the :meth:`verify_decorated_class`
        method to ensure that this does not happen. These warnings can be ignored
        by the user with the ``ignore_methods`` or ``ignore_all`` parameters.

    :Example:
        >>> from vanguard.base import GPController
        >>>
        >>> @Decorator(framework_class=GPController, required_decorators=set())
        ... class NewGPController(GPController):
        ...     pass
    """

    def __init__(
        self,
        framework_class: Type[T],
        required_decorators: Iterable[Type[DecoratorType]],
        ignore_methods: Iterable[str] = (),
        ignore_all: bool = False,
        raise_instead: bool = False,
    ):
        """
        Initialise self.

        :param framework_class: All unexpected/overwritten methods are relative to this class.
        :param required_decorators: A set (or other iterable) of decorators which must have been
                applied before (i.e. below) this one.
        :param ignore_methods: If these method names are found to have been added or overwritten,
                then an error or warning will not be raised.
        :param ignore_all: If True, all unexpected/overwritten methods will be ignored.
        :param raise_instead: If True, unexpected/overwritten methods will raise errors
                instead of emitting warnings.
        """
        self.framework_class = framework_class
        self.required_decorators = set(required_decorators)
        self.ignore_methods = ignore_methods
        self.ignore_all = ignore_all
        self.raise_instead = raise_instead

    def __call__(self, cls: Type[T]) -> Type[T]:
        self.verify_decorated_class(cls)
        decorated_class = self._decorate_class(cls)
        if decorated_class is not cls:
            decorated_class.__decorators__ = cls.__decorators__ + [type(self)]
        return decorated_class

    def _decorate_class(self, cls: Type[T]) -> Type[T]:
        """Return a wrapped version of a class."""
        return cls

    def verify_decorated_class(self, cls: Type[T]) -> None:
        """
        Verify that a class can be decorated by this instance.

        :param cls: The class to be decorated.
        :raises TypeError: If cls is not a subclass of the framework_class.
        """
        if not issubclass(cls, self.framework_class):
            raise TypeError(f"Can only apply decorator to subclasses of {self.framework_class.__name__}.")

        __decorators__ = getattr(cls, "__decorators__", [])

        if __decorators__:
            latest_decorator_class = __decorators__[-1]
            if issubclass(latest_decorator_class, TopMostDecorator):
                raise errors.TopmostDecoratorError("Cannot decorate this class!")

        missing_decorators = self.required_decorators - set(__decorators__)
        if missing_decorators:
            raise errors.MissingRequirementsError(f"The following decorators are missing: {repr(missing_decorators)}")

        if not self.ignore_all:
            super_methods = {key for key, value in getmembers(self.framework_class) if isfunction(value)}
            potentially_invalid_classes = [
                other_class for other_class in reversed(cls.__mro__) if other_class not in self.framework_class.__mro__
            ]
            for other_class in potentially_invalid_classes:
                self._verify_class_has_no_newly_added_methods(other_class, super_methods)

    def _verify_class_has_no_newly_added_methods(self, cls: Type[T], super_methods: Set[str]) -> None:
        """
        Verify that a class has not overwritten methods in the framework class or declared any new ones.

        :param cls: The class to be checked.
        :param super_methods: A set of method names found in the framework class.
        :raises errors.UnexpectedMethodError: If an unexpected method is found, and the
            :attr:`vanguard.decoratorutils.basedecorator.Decorator.raise_instead` is ``True``.
        :raises errors.OverwrittenMethodError: If a method has been overwritten, and the
            :attr:`vanguard.decoratorutils.basedecorator.Decorator.raise_instead` is ``True``.
        """
        cls_methods = {key for key, value in getmembers(cls) if isfunction(value)}
        ignore_methods = set(self.ignore_methods) | {"__wrapped__"}

        extra_methods = cls_methods - super_methods - ignore_methods
        if extra_methods:
            message = f"The class {cls.__name__!r} has added the following unexpected methods: {extra_methods!r}."
            if self.raise_instead:
                raise errors.UnexpectedMethodError(message)
            else:
                warnings.warn(message, errors.UnexpectedMethodWarning)

        overwritten_methods = {method for method in cls_methods if method in cls.__dict__} - ignore_methods
        if overwritten_methods:
            message = f"The class {cls.__name__!r} has overwritten the following methods: {overwritten_methods!r}."
            if self.raise_instead:
                raise errors.OverwrittenMethodError(message)
            else:
                warnings.warn(message, errors.OverwrittenMethodWarning)


class TopMostDecorator(Decorator):
    """
    A specific decorator which cannot be decorated.

    Top-most decorators are intended to be just that -- decorators which are at
    the top of the stack.  This is often a last resort, when it doesn't make
    sense to add any more functionality, and should be used sparingly.

    :Example:
        >>> from typing import Type, TypeVar
        >>>
        >>> from vanguard.base import GPController
        >>> from vanguard.decoratorutils import wraps_class
        >>>
        >>> ControllerType = TypeVar('ControllerType', bound=GPController)
        >>>
        >>> class MyDecorator(Decorator):
        ...     def _decorate_class(self, cls: Type[ControllerType]) -> Type[ControllerType]:
        ...         @wraps_class(cls)
        ...         class InnerClass(cls):
        ...             pass
        ...         return InnerClass
        >>>
        >>> class MyTopMostDecorator(TopMostDecorator):
        ...     def _decorate_class(self, cls: Type[ControllerType]) -> Type[ControllerType]:
        ...         @wraps_class(cls)
        ...         class InnerClass(cls):
        ...             pass
        ...         return InnerClass
        >>>
        >>> @MyTopMostDecorator(framework_class=GPController, required_decorators={})
        ... @MyDecorator(framework_class=GPController, required_decorators={})
        ... class MyController(GPController):
        ...     pass
        >>>
        >>> @MyDecorator(framework_class=GPController, required_decorators={})  # doctest: +ELLIPSIS
        ... @MyTopMostDecorator(framework_class=GPController, required_decorators={})
        ... class MyController(GPController):
        ...     pass
        Traceback (most recent call last):
            ...
        vanguard.decoratorutils.errors.TopmostDecoratorError: Cannot decorate this class!
    """

    pass
