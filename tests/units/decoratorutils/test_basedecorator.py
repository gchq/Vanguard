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

"""Contains tests for the decorators."""

import contextlib
from typing import Any, TypeVar, Union

import pytest

from tests.cases import assert_not_warns
from vanguard.base import GPController
from vanguard.decoratorutils import Decorator, TopMostDecorator, errors, wraps_class
from vanguard.decoratorutils.errors import TopmostDecoratorError
from vanguard.vanilla import GaussianGPController

ControllerT = TypeVar("ControllerT", bound=GPController)
T = TypeVar("T")


class SimpleNumber:
    """An example class for a real number."""

    __decorators__ = []

    def __init__(self, number: Union[int, float]) -> None:
        self.number = number

    def add_5(self) -> Union[int, float]:
        """Add 5 to a number."""
        return self.number + 5

    def function_with_no_annotations_or_docstring(self, t):  # noqa: D102
        # This function deliberately has no docstring, to check that one is not somehow added by the decorator.
        return [t, t]


class SquareResult(Decorator):
    """Square the result of a SimpleNumber class."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework_class=SimpleNumber, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: type[SimpleNumber]) -> type[SimpleNumber]:
        @wraps_class(cls)
        class InnerClass(cls):
            """A wrapper for normalising y inputs and variance."""

            # Note: we intentionally don't annotate this function here; this is to test that the annotations from the
            # base `SimpleNumber` class are copied over correctly.
            def add_5(self):
                """Square the result of this method."""
                result = super().add_5()
                return result**2

            def do_nothing(self) -> None:
                """Do nothing. Note that this method is not on the base class."""

            def function_with_no_annotations_or_docstring(self, t: T) -> list[T]:
                """
                Return three of whatever was passed in.

                The function actually has annotations and a docstring here! However, they'll be overwritten by
                :func:`wraps_class`.
                """
                return [t, t, t]

        return InnerClass


class RequiresSquareResult(Decorator):
    """A decorator with a requirement."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework_class=SimpleNumber, required_decorators={SquareResult}, **kwargs)


class DummyDecorator(Decorator):
    """A dummy `Decorator` used for testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: type[ControllerT]) -> type[ControllerT]:
        @wraps_class(cls)
        class InnerClass(cls):
            """A subclass which adds no functionality."""

        return super()._decorate_class(InnerClass)


class OtherDummyDecorator(Decorator):
    """A dummy `Decorator` used for testing. Identical to `DummyDecorator` in all but name."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: type[ControllerT]) -> type[ControllerT]:
        @wraps_class(cls)
        class InnerClass(cls):
            """A subclass which adds no functionality."""

        return super()._decorate_class(InnerClass)


class DummySubclassDecorator(DummyDecorator):
    """A cooperative subclass of `DummyDecorator`, used for testing."""

    def _decorate_class(self, cls: type[ControllerT]) -> type[ControllerT]:
        super_decorated = super()._decorate_class(cls)

        @wraps_class(super_decorated)
        class InnerClass(super_decorated):
            """A subclass which adds no functionality."""

        return super()._decorate_class(InnerClass)


class DummyTopmostDecorator(TopMostDecorator):
    """A dummy `TopmostDecorator` used for testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: type[ControllerT]) -> type[ControllerT]:
        @wraps_class(cls)
        class InnerClass(cls):
            """A subclass which adds no functionality."""

        return super()._decorate_class(InnerClass)


class TestDecoratorList:
    """Testing the __decorators__ attribute."""

    def test_initial_value(self) -> None:
        """On an undecorated controller, `__decorators__` should be empty."""
        assert not GPController.__decorators__

    def test_once_decorated_value(self) -> None:
        """On a decorated controller, `__decorators__` should contain each decorator that decorates the controller."""

        @DummyDecorator()
        class DecoratedController(GPController):
            """Decorated GP controller class for testing."""

        assert DecoratedController.__decorators__ == [DummyDecorator]

    def test_twice_decorated_value(self) -> None:
        """On a decorated controller, `__decorators__` should contain each decorator that decorates the controller."""

        @DummyDecorator()
        @OtherDummyDecorator()
        class DoubleDecoratedController(GPController):
            """Decorated GP controller class for testing."""

        # note that the order here is the order the decorators are actually applied in - lowest first
        assert DoubleDecoratedController.__decorators__ == [OtherDummyDecorator, DummyDecorator]

    def test_subclass_decorated_value(self) -> None:
        """
        Check that when decorated once, `__decorators__` only contains one decorator.

        This test is to check for a possible failure mode when using subclasses - even if a cooperative subclass of a
        decorator calls `super()._decorate_class()` in its own implementation of `_decorate_class`, we want to ensure
        that only one decorator is added to `__decorators`.
        """

        @DummySubclassDecorator()
        class SubclassDecoratedController(GPController):
            """Decorated GP controller class for testing."""

        assert SubclassDecoratedController.__decorators__ == [DummySubclassDecorator]


class TestWrapping:
    """Tests for the `wraps_class` decorator."""

    def test_answer(self) -> None:
        """Test that the decorator modifies the `add_5` function as intended."""

        @SquareResult()
        class DecoratedNumber(SimpleNumber):
            """Decorated class for testing."""

        number = DecoratedNumber(10)
        assert 15**2 == number.add_5()

    @pytest.mark.parametrize("attr", ["__doc__", "__name__", "__qualname__", "__annotations__"])
    @pytest.mark.parametrize("which", ["self", "__init__", "add_5", "function_with_no_annotations_or_docstring"])
    @pytest.mark.parametrize("where", ["class", "instance"])
    def test_dunder_attributes_equal(self, attr: str, where: str, which: str):
        """Test that dunder attributes are correctly copied from the decorated class."""
        if where == "class":
            before = SimpleNumber
            after = SquareResult()(SimpleNumber)
        elif where == "instance":
            before = SimpleNumber(10)
            after = SquareResult()(SimpleNumber)(10)
        else:
            raise ValueError(where)

        if which != "self":
            # i.e. if `which` is "__init__" or "add_5" etc
            before = getattr(before, which)
            after = getattr(after, which)

        # check that the attribute is set on one iff it's set on the other
        assert hasattr(before, attr) == hasattr(after, attr)
        if hasattr(before, attr):
            # and if it is set, check that it's equal
            assert getattr(before, attr) == getattr(after, attr)

    def test_wrapped_attribute(self) -> None:
        """Test that the `__wrapped__` attribute is set correctly."""
        assert SquareResult()(SimpleNumber).__wrapped__ is SimpleNumber

    def test_decoration_changes_class(self) -> None:
        """Test that the decorated class is different from the original class."""
        assert SquareResult()(SimpleNumber) is not SimpleNumber

    def test_new_methods_are_not_wrapped(self):
        """Test that `wraps_class` only replaces attributes on methods that exist on the base class."""
        assert SquareResult()(SimpleNumber).do_nothing.__doc__ == (
            "Do nothing. Note that this method is not on the base class."
        )


class TestErrorsWhenOverwriting:
    """Testing breaking the decorator by overwriting or extending parts of the base class."""

    @pytest.mark.parametrize("mode", ["new_method", "override_method"])
    @pytest.mark.parametrize("raise_instead", [True, False])
    @pytest.mark.parametrize("ignore", ["specific", "all", "other", "none"])
    @pytest.mark.parametrize("who_is_wrong", ["subclass", "superclass", "both"])
    def test_unexpected_method_change(self, *, mode: str, ignore: str, raise_instead: bool, who_is_wrong: str):
        """
        Test what happens if the decorated class's methods are different from those of the framework class.

        We test in two modes:
         - `new_method`: Adding a new method not present on the base class
         - `override_method`: Overriding a method that was present on the base class

        We test with the `raise_instead` flag set to both true and false, testing for an exception or warning as
        appropriate.

        We test with three "ignore" settings:
         - Don't do any ignores (and check that the correct warning/error is raised)
         - Specify some other name to be ignored (and check the correct warning/error is still raised)
         - Ignore by specifying the name (and check no error/warning is raised)
         - Ignore by setting `ignore_all = True` (and check no error/warning is raised)

        We test with the error in three locations:
         - Unexpected method change is in the class being decorated (class being decorated is blamed)
         - Unexpected method change is in a superclass of the class being decorated (superclass is blamed)
         - Both the class being decorated and a superclass have an unexpected method change (superclass is blamed)
        """
        # Set up the classes.
        if who_is_wrong == "subclass":

            class MiddleNumber(SimpleNumber):
                """Transparent subclass."""
        else:
            if mode == "new_method":

                class MiddleNumber(SimpleNumber):
                    """Subclass that adds a new method."""

                    def something_new(self) -> None:
                        pass
            elif mode == "override_method":

                class MiddleNumber(SimpleNumber):
                    """Subclass that overrides an existing method."""

                    def add_5(self) -> Union[int, float]:
                        return super().add_5() + 1
            else:
                raise ValueError("mode")

        if who_is_wrong == "superclass":

            class NewNumber(MiddleNumber):
                """Transparent subclass."""
        else:
            if mode == "new_method":

                class NewNumber(MiddleNumber):
                    """Subclass that adds a new method."""

                    def something_new(self) -> None:
                        pass
            elif mode == "override_method":

                class NewNumber(MiddleNumber):
                    """Subclass that overrides an existing method."""

                    def add_5(self) -> Union[int, float]:
                        return super().add_5() + 1
            else:
                raise ValueError("mode")

        # Set up the test itself.
        kwargs = {"raise_instead": raise_instead}
        if ignore == "none" or ignore == "other":
            if ignore == "other":
                kwargs["ignore_methods"] = ["some_other_method"]

            # If we're not ignoring the errors, then specify what the expected error/warning is.
            if who_is_wrong == "subclass":
                blame_class = "NewNumber"
            elif who_is_wrong == "superclass" or who_is_wrong == "both":
                blame_class = "MiddleNumber"
            else:
                raise ValueError(who_is_wrong)

            if mode == "new_method":
                expected_message = (
                    f"{SquareResult.__name__!r}: The class '{blame_class}' has added the following unexpected methods"
                )
                expected_types = errors.UnexpectedMethodError, errors.UnexpectedMethodWarning
            elif mode == "override_method":
                expected_message = (
                    f"{SquareResult.__name__!r}: The class '{blame_class}' has overwritten the following methods"
                )
                expected_types = errors.OverwrittenMethodError, errors.OverwrittenMethodWarning
            else:
                raise ValueError(mode)

            if raise_instead:
                context = pytest.raises(expected_types[0], match=expected_message)
            else:
                context = pytest.warns(expected_types[1], match=expected_message)
        elif ignore == "specific":
            # Ignore by setting `ignore_methods`.
            if mode == "new_method":
                kwargs["ignore_methods"] = ["something_new"]
            elif mode == "override_method":
                kwargs["ignore_methods"] = ["add_5"]
            else:
                raise ValueError(mode)
            context = contextlib.nullcontext() if raise_instead else assert_not_warns()
        elif ignore == "all":
            # Ignore by setting `ignore_all`.
            kwargs["ignore_all"] = True
            context = contextlib.nullcontext() if raise_instead else assert_not_warns()
        else:
            raise ValueError(ignore)

        # Actually perform the decoration, and check for errors/warnings as appropriate
        with context:
            SquareResult(**kwargs)(NewNumber)


class TestInvalidDecoration:
    """Test that decorators do not allow invalid classes to be decorated."""

    def test_can_decorate_correct_framework_class(self):
        """Test that we can decorate subclasses of the framework class without error."""
        DummyDecorator()(GaussianGPController)

    def test_cannot_decorate_incorrect_framework_class(self):
        """Test that we can decorate subclasses of the framework class without error."""
        with pytest.raises(TypeError, match=f"Can only apply decorator to subclasses of {GPController.__name__}"):
            DummyDecorator()(object)

    def test_topmost_decorator_error(self):
        """Test that we get an appropriate error if we try to decorate on top of a `TopmostDecorator`."""
        # decorating once is fine
        decorated_once = DummyTopmostDecorator()(GaussianGPController)
        # but twice raises an error
        with pytest.raises(TopmostDecoratorError):
            DummyDecorator()(decorated_once)

    def test_missing_requirements(self) -> None:
        """Test that we get an appropriate error if we are missing requirements for a decorator."""
        with pytest.raises(errors.MissingRequirementsError, match=SquareResult.__name__):

            @RequiresSquareResult()
            class NewNumber(SimpleNumber):  # pylint: disable=unused-variable
                """Declaring this class should throw an error, as we're missing a required decorator."""

    def test_passed_requirements(self) -> None:
        """Test that when all requirements for a decorator are satisfied, no error is thrown."""

        @RequiresSquareResult(ignore_methods=("__init__", "add_5"))
        @SquareResult()
        class NewNumber(SimpleNumber):  # pylint: disable=unused-variable
            """Declaring this class should not throw any error, as all requirements are satisfied."""
