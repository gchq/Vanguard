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

from typing import Any, Type, TypeVar, Union

import pytest

from tests.cases import VanguardTestCase
from vanguard.base import GPController
from vanguard.decoratorutils import Decorator, TopMostDecorator, errors, wraps_class
from vanguard.decoratorutils.errors import TopmostDecoratorError
from vanguard.vanilla import GaussianGPController

ControllerT = TypeVar("ControllerT", bound=GPController)


class SimpleNumber:
    """An example class for a real number."""

    __decorators__ = []

    def __init__(self, number: Union[int, float]) -> None:
        self.number = number

    def add_5(self) -> Union[int, float]:
        """Add 5 to a number."""
        return self.number + 5


class SquareResult(Decorator):
    """Square the result of a SimpleNumber class."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework_class=SimpleNumber, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: Type[SimpleNumber]) -> Type[SimpleNumber]:
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

        return InnerClass


class DummyDecorator(Decorator):
    """A dummy `Decorator` used for testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        @wraps_class(cls)
        class InnerClass(cls):
            """A subclass which adds no functionality."""

        return super()._decorate_class(InnerClass)


class OtherDummyDecorator(Decorator):
    """A dummy `Decorator` used for testing. Identical to `DummyDecorator` in all but name."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        @wraps_class(cls)
        class InnerClass(cls):
            """A subclass which adds no functionality."""

        return super()._decorate_class(InnerClass)


class DummySubclassDecorator(DummyDecorator):
    """A cooperative subclass of `DummyDecorator`, used for testing."""

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
        super_decorated = super()._decorate_class(cls)

        @wraps_class(super_decorated)
        class InnerClass(super_decorated):
            """A subclass which adds no functionality."""

        return super()._decorate_class(InnerClass)


class DummyTopmostDecorator(TopMostDecorator):
    """A dummy `TopmostDecorator` used for testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: Type[ControllerT]) -> Type[ControllerT]:
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
    @pytest.mark.parametrize("which", ["class", "instance", "method-on-class", "method-on-instance", "class-init"])
    def test_dunder_attributes_equal(self, attr: str, which: str):
        """Test that dunder attributes are correctly copied from the decorated class."""
        if which == "class":
            before = SimpleNumber
            after = SquareResult()(SimpleNumber)
        elif which == "instance":
            before = SimpleNumber(10)
            after = SquareResult()(SimpleNumber)(10)
        elif which == "method-on-class":
            before = SimpleNumber.add_5
            after = SquareResult()(SimpleNumber).add_5
        elif which == "class-init":
            before = SimpleNumber.__init__
            after = SquareResult()(SimpleNumber).__init__
        elif which == "method-on-instance":
            before = SimpleNumber(10).add_5
            after = SquareResult()(SimpleNumber)(10).add_5
        else:
            raise ValueError(which)

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


class TestErrorsWhenOverwriting(VanguardTestCase):
    """Testing breaking the decorator by overwriting or extending parts of the base class."""

    def test_overwrite_method_with_raise(self) -> None:
        """Test that overwriting a method throws an error instead."""
        expected_error_message = "The class 'NewNumber' has overwritten the following methods: {'add_5'}."
        with pytest.raises(errors.OverwrittenMethodError, match=expected_error_message):

            @SquareResult(raise_instead=True)
            class NewNumber(SimpleNumber):  # pylint: disable=unused-variable
                """Declaring this class should throw an error."""

                def add_5(self) -> Union[int, float]:
                    return super().add_5() + 1

    def test_overwrite_method(self) -> None:
        """Test that overwriting a method throws a warning."""
        expected_warning_message = "The class 'NewNumber' has overwritten the following methods: {'add_5'}."
        with pytest.warns(errors.OverwrittenMethodWarning, match=expected_warning_message):

            @SquareResult()
            class NewNumber(SimpleNumber):  # pylint: disable=unused-variable
                """Declaring this class should raise a warning."""

                def add_5(self) -> Union[int, float]:
                    return super().add_5() + 1

    def test_overwrite_method_with_ignore(self) -> None:
        """Test that such an error can be avoided."""
        with self.assertNotWarns(errors.OverwrittenMethodWarning):

            @SquareResult(ignore_methods=("add_5",))
            class NewNumber(SimpleNumber):  # pylint: disable=unused-variable
                """Declaring this class should not raise any OverwrittenMethodWarning."""

                def add_5(self) -> Union[int, float]:
                    return super().add_5() + 1

    def test_unexpected_method_with_raise(self) -> None:
        """Test that creating a new method throws an error instead."""
        expected_error_message = "The class 'NewNumber' has added the following unexpected methods: {'something_new'}."
        with pytest.raises(errors.UnexpectedMethodError, match=expected_error_message):

            @SquareResult(raise_instead=True)
            class NewNumber(SimpleNumber):  # pylint: disable=unused-variable
                """Declaring this class should throw an error."""

                def something_new(self) -> None:
                    pass

    def test_unexpected_method_with_warn(self) -> None:
        """Test that creating a new method throws a warning."""
        expected_warning_message = (
            "The class 'NewNumber' has added the following unexpected methods: {'something_new'}."
        )
        with pytest.warns(errors.UnexpectedMethodWarning, match=expected_warning_message):

            @SquareResult()
            class NewNumber(SimpleNumber):  # pylint: disable=unused-variable
                """Declaring this class should raise a warning."""

                def something_new(self) -> None:
                    pass

    def test_unexpected_method_with_ignore(self) -> None:
        """Test that such an error can be avoided."""
        with self.assertNotWarns(errors.UnexpectedMethodWarning):

            @SquareResult(ignore_methods=["something_new"])
            class NewNumber(SimpleNumber):  # pylint: disable=unused-variable
                """Declaring this class should not raise UnexpectedMethodWarning."""

                def something_new(self) -> None:
                    pass

    def test_overwrite_method_with_superclass_subclass_wrong(self) -> None:
        """Warning should reference the SUBCLASS."""
        expected_warning_message = "The class 'NewNumber' has overwritten the following methods: {'add_5'}."

        class MiddleNumber(SimpleNumber):
            """An intermediate number."""

        with pytest.warns(errors.OverwrittenMethodWarning, match=expected_warning_message):

            @SquareResult()
            class NewNumber(MiddleNumber):  # pylint: disable=unused-variable
                """Declaring this class should raise a warning."""

                def add_5(self) -> Union[int, float]:
                    return super().add_5() + 1

    def test_overwrite_method_with_superclass_superclass_wrong(self) -> None:
        """Warning should reference the SUPERCLASS."""
        expected_warning_message = "The class 'MiddleNumber' has overwritten the following methods: {'add_5'}."

        class MiddleNumber(SimpleNumber):
            """An intermediate number."""

            def add_5(self) -> Union[int, float]:
                return super().add_5() + 1

        with pytest.warns(errors.OverwrittenMethodWarning, match=expected_warning_message):

            @SquareResult()
            class NewNumber(MiddleNumber):  # pylint: disable=unused-variable
                """Declaring this class should raise a warning."""

    def test_overwrite_method_with_superclass_both_wrong(self) -> None:
        """Warning should reference the SUPERCLASS."""
        expected_warning_message = "The class 'MiddleNumber' has overwritten the following methods: {'add_5'}."

        class MiddleNumber(SimpleNumber):
            """An intermediate number."""

            def add_5(self) -> Union[int, float]:
                return super().add_5() + 1

        with pytest.warns(errors.OverwrittenMethodWarning, match=expected_warning_message):

            @SquareResult()
            class NewNumber(MiddleNumber):  # pylint: disable=unused-variable
                """Declaring this class should raise a warning."""

                def add_5(self) -> Union[int, float]:
                    return super().add_5() + 2

    def test_missing_requirements(self) -> None:
        """Test that we get an appropriate error if we are missing requirements for a decorator."""

        class RequirementDecorator(Decorator):
            """A decorator with a requirement."""

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(framework_class=SimpleNumber, required_decorators={SquareResult}, **kwargs)

        with pytest.raises(errors.MissingRequirementsError, match=SquareResult.__name__):

            @RequirementDecorator()
            class NewNumber(SimpleNumber):  # pylint: disable=unused-variable
                """Declaring this class should throw an error, as we're missing a required decorator"""

    def test_passed_requirements(self) -> None:
        """Test that when all requirements for a decorator are satisfied, no error is thrown."""

        class RequirementDecorator(Decorator):
            """A decorator with a requirement."""

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(framework_class=SimpleNumber, required_decorators={SquareResult}, **kwargs)

        @RequirementDecorator(ignore_methods=("__init__", "add_5"))
        @SquareResult()
        class NewNumber(SimpleNumber):  # pylint: disable=unused-variable
            """Declaring this class should not throw any error, as all requirements are satisfied."""


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
