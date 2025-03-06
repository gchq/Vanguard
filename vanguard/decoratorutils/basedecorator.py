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
Contains the BaseDecorator class.
"""

import warnings
from collections.abc import Iterable
from inspect import getmembers, isfunction
from typing import Optional, TypeVar

from vanguard.decoratorutils import errors

T = TypeVar("T")
DecoratorT = TypeVar("DecoratorT", bound="Decorator")


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
        framework_class: type[T],
        required_decorators: Iterable[type[DecoratorT]],
        ignore_methods: Iterable[str] = (),
        ignore_all: bool = False,
        raise_instead: bool = False,
    ) -> None:
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

    def __call__(self, cls: type[T]) -> type[T]:
        """
        Decorate a class, checking that the class is appropriate before decorating.

        :param cls: The class to decorate.
        :return: The decorated class.
        :raises TypeError: If cls is not a subclass of the framework_class.
        :raises TopmostDecoratorError: If cls is already decorated with a
            :class:`~vanguard.decoratorutils.basedecorator.TopMostDecorator`.
        :raises MissingRequirementsError: If cls is missing a required decorator.
        """
        self.verify_decorated_class(cls)
        decorated_class = self._decorate_class(cls)
        if decorated_class is not cls:
            decorated_class.__decorators__ = decorated_class.__decorators__ + [type(self)]
        return decorated_class

    @property
    def safe_updates(self) -> dict[type, set[str]]:
        """Get a dictionary (class -> set[names]) of overrides/new methods that we consider "safe"."""
        # This import needs to be lazy to avoid circular imports.
        from vanguard.vanilla import GaussianGPController  # pylint: disable=import-outside-toplevel

        return {GaussianGPController: {"__init__"}}

    @staticmethod
    def _add_to_safe_updates(old: dict[type, set[str]], new: dict[type, set[str]]) -> dict[type, set[str]]:
        """
        Add to a safe-updates set in-place - see the example.

        :Example:
            >>> class A: pass
            >>> class B: pass
            >>> class C: pass
            >>> old = {A: {"1", "2"}, B: {"other"}}
            >>> new = {A: {"3"}, C: {"newer"}}
            >>> updated = Decorator._add_to_safe_updates(old, new)
            >>> sorted(updated[A])  # sets are updated
            ['1', '2', '3']
            >>> updated[B]  # non-overlapping keys are both added
            {'other'}
            >>> updated[C]
            {'newer'}
            >>> updated == old  # `old` is modified in-place
            True
        """
        for key, value in new.items():
            if key in old:
                old[key].update(value)
            else:
                old[key] = value
        return old

    def _decorate_class(self, cls: type[T]) -> type[T]:
        """Return a wrapped version of a class."""
        return cls

    def verify_decorated_class(self, cls: type[T]) -> None:
        """
        Verify that a class can be decorated by this instance.

        :param cls: The class to be decorated.
        :raises TypeError: If cls is not a subclass of the framework_class.
        :raises TopmostDecoratorError: If cls is already decorated with a
            :class:`~vanguard.decoratorutils.basedecorator.TopMostDecorator`.
        :raises MissingRequirementsError: If cls is missing a required decorator.
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
            raise errors.MissingRequirementsError(
                f"The following decorators are missing for decorator {type(self).__name__}: {repr(missing_decorators)}"
            )

        if not self.ignore_all:
            super_methods = {key for key, value in getmembers(self.framework_class) if isfunction(value)}
            potentially_invalid_classes = [
                other_class for other_class in reversed(cls.__mro__) if other_class not in self.framework_class.__mro__
            ]
            for other_class in potentially_invalid_classes:
                self._verify_class_has_no_newly_added_methods(other_class, super_methods)

    def _verify_class_has_no_newly_added_methods(self, cls: type[T], super_methods: set[str]) -> None:
        """
        Verify that a class has not overwritten methods in the framework class or declared any new ones.

        :param cls: The class to be checked.
        :param super_methods: A set of method names found in the framework class.
        :raises errors.UnexpectedMethodError: If an unexpected method is found, and the
            :attr:`vanguard.decoratorutils.basedecorator.Decorator.raise_instead` is ``True``.
        :raises errors.OverwrittenMethodError: If a method has been overwritten, and the
            :attr:`vanguard.decoratorutils.basedecorator.Decorator.raise_instead` is ``True``.
        """
        cls_methods = {
            key
            for key, value in getmembers(cls, isfunction)
            if key not in self.safe_updates.get(self._get_method_implementation(cls, key), set())
        }
        ignore_methods = set(self.ignore_methods) | {"__wrapped__"}

        extra_methods = cls_methods - super_methods - ignore_methods
        if extra_methods:
            if __debug__:
                extra_method_messages = "\n".join(
                    f"* {getattr(cls, method).__vanguard_wrap_source__.__class__.__module__}"
                    f".{getattr(cls, method).__vanguard_wrap_source__.__class__.__qualname__}.{method} "
                    f"(wrapping {getattr(cls, method).__module__}.{getattr(cls, method).__qualname__})"
                    if hasattr(getattr(cls, method), "__vanguard_wrap_source__")
                    else f"* {getattr(cls, method).__module__}.{getattr(cls, method).__qualname__}"
                    for method in extra_methods
                )
                message = (
                    f"{self.__class__.__name__!r}: The class {cls.__name__!r} "
                    f"has added the following unexpected methods: \n"
                    f"{extra_method_messages}"
                )
            else:
                message = "Unexpected methods added to the class"
            if self.raise_instead:
                raise errors.UnexpectedMethodError(message)
            else:
                warnings.warn(message, errors.UnexpectedMethodWarning, stacklevel=4)

        overwritten_methods = {method for method in cls_methods if method in cls.__dict__} - ignore_methods
        if overwritten_methods:
            if __debug__:
                overwritten_method_messages = "\n".join(
                    f"* {getattr(cls, method).__vanguard_wrap_source__.__class__.__module__}"
                    f".{getattr(cls, method).__vanguard_wrap_source__.__class__.__qualname__}.{method} "
                    f"(wrapping {getattr(cls, method).__module__}.{getattr(cls, method).__qualname__})"
                    if hasattr(getattr(cls, method), "__vanguard_wrap_source__")
                    else f"* {getattr(cls, method).__module__}.{getattr(cls, method).__qualname__}"
                    for method in overwritten_methods
                )
                message = (
                    f"{self.__class__.__name__!r}: The class {cls.__name__!r} "
                    f"has overwritten the following methods: \n"
                    f"{overwritten_method_messages}"
                )
            else:
                message = "Unexpected methods overwritten by the class"
            if self.raise_instead:
                raise errors.OverwrittenMethodError(message)
            else:
                warnings.warn(message, errors.OverwrittenMethodWarning, stacklevel=4)

    @staticmethod
    def _get_method_implementation(subclass: type, method_name: str) -> Optional[type]:
        """
        Get the class that provides the implementation of a method.

        Has a special case for classes decorated with @wraps_class.

        This is doing some rather finicky introspection, so might end up being quite fragile.

        In particular, things that might cause problems are:

          - Classes that inherit from a wrapped class.
          - Nested classes.

        :Example:
            >>> from vanguard.vanilla import GaussianGPController
            >>> Decorator._get_method_implementation(GaussianGPController, "__init__").__name__
            'GaussianGPController'
            >>> Decorator._get_method_implementation(GaussianGPController, "fit").__name__
            'GPController'
            >>> Decorator._get_method_implementation(GaussianGPController, "_predictive_likelihood").__name__
            'BaseGPController'

        :Example:
            >>> from vanguard.classification import BinaryClassification
            >>> from vanguard.variational import VariationalInference
            >>> from vanguard.vanilla import GaussianGPController
            >>> @BinaryClassification()
            ... @VariationalInference()
            ... class BinaryController(GaussianGPController):
            ...     pass
            >>> Decorator._get_method_implementation(BinaryController, "classify_points").__name__
            'BinaryClassification'


        :param subclass: The class that the method belongs to. (Note that this could be a subclass.)
        :param method_name: The name of the method to get the implementing class for.
        :returns: The class that provides the implementation for the method, or None if the class could not be found.
        """
        method = getattr(subclass, method_name)
        # Handle wrapping by @wraps_class
        if hasattr(method, "__vanguard_wrap_source__"):
            return getattr(method, "__vanguard_wrap_source__").__class__

        method_path = method.__qualname__.split(".")
        mro_unwrapped = {
            (c.__vanguard_wrap_source__.__class__.__name__ if hasattr(c, "__vanguard_wrap_source__") else c.__name__): c
            for c in subclass.__mro__
        }
        try:
            outer_class = mro_unwrapped[method_path[0]]
        except KeyError:
            # TODO: logging here?
            # https://github.com/gchq/Vanguard/issues/123
            return None

        if len(method_path) == 2:
            # Simple case - Class.method()
            return outer_class
        if (
            # Vanguard decorator case: DecoratorClass._decorate_class.<locals>.InnerClass.method()
            hasattr(outer_class, "__vanguard_wrap_source__")
            and isinstance(outer_class.__vanguard_wrap_source__, Decorator)
            and len(method_path) == 5
            # method_path[0] is the class name
            and method_path[1] == "_decorate_class"
            and method_path[2] == "<locals>"
            # method_path[3] is conventionally `InnerClass`, but don't enforce this
            # method_path[4] is the method_name
        ):
            return outer_class.__vanguard_wrap_source__.__class__
        else:
            # TODO: logging here?
            # https://github.com/gchq/Vanguard/issues/123
            return None


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
