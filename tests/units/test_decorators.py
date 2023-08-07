"""
Contains tests for the decorators.
"""
import inspect
import unittest

from vanguard.base import GPController
from vanguard.decoratorutils import Decorator, errors, process_args, wraps_class


class DummyDecorator1(Decorator):
    """
    A dummy decorator used for testing.
    """
    def __init__(self, **kwargs):
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls):
        @wraps_class(cls)
        class InnerClass(cls):
            """A subclass which adds no functionality."""
            pass

        return super()._decorate_class(InnerClass)


class DummyDecorator2(Decorator):
    """
    A dummy decorator used for testing.
    """
    def __init__(self, **kwargs):
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls):
        @wraps_class(cls)
        class InnerClass(cls):
            """A subclass which adds no functionality."""
            pass

        return super()._decorate_class(InnerClass)


class DummySubclass(DummyDecorator1):
    """
    A subclass of a decorator.
    """
    def _decorate_class(self, cls):
        super_decorated = super()._decorate_class(cls)

        @wraps_class(super_decorated)
        class InnerClass(super_decorated):
            """A subclass which adds no functionality."""
            pass

        return super()._decorate_class(InnerClass)


class TrackingTests(unittest.TestCase):
    """
    Testing the __decorators__ attribute.
    """
    def setUp(self):
        """Code to run before each test."""
        @DummyDecorator1()
        class DecoratedOnceGPController(GPController):
            """A dummy decorated class."""
            pass

        @DummyDecorator2()
        @DummyDecorator1()
        class DecoratedTwiceGPController(GPController):
            """A dummy decorated class."""
            pass

        @DummySubclass()
        class DecoratedWithSubclassController(GPController):
            """A dummy class decorated with a controller."""
            pass

        self.decorated_once_controller_class = DecoratedOnceGPController
        self.decorated_twice_controller_class = DecoratedTwiceGPController
        self.decorated_with_subclass_controller = DecoratedWithSubclassController

    def test_initial_value(self):
        """List should be empty."""
        self.assertListEqual([], GPController.__decorators__)

    def test_once_decorated_value(self):
        """List should contain the decorator."""
        self.assertListEqual([DummyDecorator1], self.decorated_once_controller_class.__decorators__)

    def test_twice_decorated_value(self):
        """List should contain both decorator."""
        self.assertListEqual([DummyDecorator1, DummyDecorator2], self.decorated_twice_controller_class.__decorators__)

    def test_subclass_decorated_value(self):
        """Should only contain one decorator."""
        self.assertEqual([DummySubclass], self.decorated_with_subclass_controller.__decorators__)


class AttributeTests(unittest.TestCase):
    """
    Tests that the attributes have been properly updated.
    """
    def setUp(self):
        """Code to run before each test."""
        class SimpleNumber:
            """
            An example class for a real number.
            """
            __decorators__ = []

            def __init__(self, number):
                self.number = number

            def add_5(self):
                """Add 5 to a number."""
                return self.number + 5

        class SquareResult(Decorator):
            """
            Square the result of a SimpleNumber class.
            """
            def __init__(self, **kwargs):
                super().__init__(framework_class=SimpleNumber, required_decorators={}, **kwargs)

            def _decorate_class(self, cls):
                @wraps_class(cls)
                class InnerClass(cls):
                    """
                    A wrapper for normalising y inputs and variance.
                    """
                    def __init__(self, *args, **kwargs):
                        """Inner initialisation."""
                        super().__init__(*args, **kwargs)

                    def add_5(self):
                        """Square the result of this method."""
                        result = super().add_5()
                        return result ** 2

                return InnerClass

        class SimilarNumber(SimpleNumber):
            """
            A superfluous subclass.
            """
            pass

        self.SimilarNumberBefore = SimilarNumber
        self.SimilarNumberAfter = SquareResult()(SimilarNumber)
        self.number = self.SimilarNumberAfter(10)

    def test_answer(self):
        """Test that the subclass has actually been applied."""
        self.assertEqual(15 ** 2, self.number.add_5())

    def test_docstrings_of_class(self):
        """Test that the class docstrings are all correct."""
        self.assertEqual(self.SimilarNumberBefore.__doc__, self.SimilarNumberAfter.__doc__)

    def test_docstrings_of_instance(self):
        """Test that the instance docstrings are all correct."""
        self.assertEqual(self.SimilarNumberBefore.__doc__, self.number.__doc__)
        self.assertEqual(self.SimilarNumberBefore.add_5.__doc__, self.number.add_5.__doc__)

    def test_names_of_class(self):
        """Test that the class names are all correct."""
        self.assertEqual(self.SimilarNumberBefore.__name__, self.SimilarNumberAfter.__name__)
        self.assertEqual(self.SimilarNumberBefore.add_5.__name__, self.SimilarNumberAfter.add_5.__name__)

    def test_qualnames_of_class(self):
        """Test that the class qualnames are all correct."""
        self.assertEqual(self.SimilarNumberBefore.__qualname__, self.SimilarNumberAfter.__qualname__)
        self.assertEqual(self.SimilarNumberBefore.add_5.__qualname__, self.SimilarNumberAfter.add_5.__qualname__)

    def test_names_of_instance(self):
        """Test that the instance names are all correct."""
        self.assertEqual(self.SimilarNumberBefore.__name__, self.number.__class__.__name__)
        self.assertEqual(self.SimilarNumberBefore.add_5.__name__, self.number.add_5.__name__)

    def test_qualnames_of_instance(self):
        """Test that the instance qualnames are all correct."""
        self.assertEqual(self.SimilarNumberBefore.__qualname__, self.number.__class__.__qualname__)
        self.assertEqual(self.SimilarNumberBefore.add_5.__qualname__, self.number.add_5.__qualname__)

    def test_wrapped_attribute(self):
        """Test that the instance names are all correct."""
        try:
            self.assertEqual(self.SimilarNumberBefore, self.SimilarNumberAfter.__wrapped__)
        except AttributeError:
            self.fail("Wrapped class does not have '__wrapped__' attribute.")


class TestErrorsWhenOverwriting(unittest.TestCase):
    """
    Testing breaking the decorator by overwriting or extending.
    """
    def setUp(self):
        """Code to run before each test."""
        class SimpleNumber:
            """
            An example class for a real number.
            """
            __decorators__ = []

            def __init__(self, number):
                self.number = number

            def add_5(self):
                """Add 5 to a number."""
                return self.number + 5

        class SquareResult(Decorator):
            """
            Square the result of a SimpleNumber class.
            """
            def __init__(self, **kwargs):
                super().__init__(framework_class=SimpleNumber, required_decorators={}, **kwargs)

            def _decorate_class(self, cls):
                @wraps_class(cls)
                class InnerClass(cls):
                    """
                    A wrapper for normalising y inputs and variance.
                    """
                    def __init__(self, *args, **kwargs):
                        """Inner initialisation."""
                        super().__init__(*args, **kwargs)

                    def add_5(self):
                        """Square the result of this method."""
                        result = super().add_5()
                        return result ** 2

                return InnerClass

        self.SimpleNumber = SimpleNumber
        self.SquareResult = SquareResult

    def test_overwrite_method_with_raise(self):
        """Test that overwriting a method throws an error instead."""
        expected_error_message = "The class 'NewNumber' has overwritten the following methods: {'add_5'}."
        with self.assertRaisesRegex(errors.OverwrittenMethodError, expected_error_message):
            @self.SquareResult(raise_instead=True)
            class NewNumber(self.SimpleNumber):
                """
                Declaring this class should throw an error.
                """
                def add_5(self):
                    return super().add_5()

    def test_overwrite_method(self):
        """Test that overwriting a method throws a warning."""
        expected_error_message = "The class 'NewNumber' has overwritten the following methods: {'add_5'}."
        with self.assertWarnsRegex(errors.OverwrittenMethodWarning, expected_error_message):
            @self.SquareResult()
            class NewNumber(self.SimpleNumber):
                """
                Declaring this class should throw an error.
                """
                def add_5(self):
                    return super().add_5()

    def test_overwrite_method_with_ignore(self):
        """Test that such an error can be avoided."""
        try:
            with self.assertWarns(errors.OverwrittenMethodWarning):
                @self.SquareResult(ignore_methods=("add_5",))
                class NewNumber(self.SimpleNumber):
                    """
                    Declaring this class should throw an error.
                    """
                    def add_5(self):
                        return super().add_5()
        except AssertionError:
            pass
        except errors.OverwrittenMethodError as error:
            self.fail(f"Should not have thrown error: {error!s}")
        else:
            self.fail("Should not have thrown warning.")

    def test_unexpected_method_with_raise(self):
        """Test that creating a new method throws an error instead."""
        expected_error_message = "The class 'NewNumber' has added the following unexpected methods: {'something_new'}."
        with self.assertRaisesRegex(errors.UnexpectedMethodError, expected_error_message):
            @self.SquareResult(raise_instead=True)
            class NewNumber(self.SimpleNumber):
                """
                Declaring this class should throw an error.
                """
                def something_new(self):
                    pass

    def test_unexpected_method_with_warn(self):
        """Test that creating a new method throws a warning."""
        expected_error_message = "The class 'NewNumber' has added the following unexpected methods: {'something_new'}."
        with self.assertWarnsRegex(errors.UnexpectedMethodWarning, expected_error_message):
            @self.SquareResult()
            class NewNumber(self.SimpleNumber):
                """
                Declaring this class should throw an error.
                """
                def something_new(self):
                    pass

    def test_unexpected_method_with_ignore(self):
        """Test that such an error can be avoided."""
        try:
            with self.assertWarns(errors.UnexpectedMethodWarning):
                @self.SquareResult(ignore_methods=["something_new"])
                class NewNumber(self.SimpleNumber):
                    """
                    Declaring this class should throw an error.
                    """
                    def something_new(self):
                        pass
        except AssertionError:
            pass
        except errors.UnexpectedMethodError as error:
            self.fail(f"Should not have thrown error: {error!s}")
        else:
            self.fail("Should not have thrown warning.")

    def test_overwrite_method_with_superclass_subclass_wrong(self):
        """Warning should reference the SUBCLASS."""
        expected_error_message = "The class 'NewNumber' has overwritten the following methods: {'add_5'}."

        class MiddleNumber(self.SimpleNumber):
            """An intermediate number."""
            pass

        with self.assertWarnsRegex(errors.OverwrittenMethodWarning, expected_error_message):
            @self.SquareResult()
            class NewNumber(MiddleNumber):
                """
                Declaring this class should throw an error.
                """
                def add_5(self):
                    return super().add_5()

    def test_overwrite_method_with_superclass_superclass_wrong(self):
        """Warning should reference the SUPERCLASS."""
        expected_error_message = "The class 'MiddleNumber' has overwritten the following methods: {'add_5'}."

        class MiddleNumber(self.SimpleNumber):
            """An intermediate number."""
            def add_5(self):
                return super().add_5()

        with self.assertWarnsRegex(errors.OverwrittenMethodWarning, expected_error_message):
            @self.SquareResult()
            class NewNumber(MiddleNumber):
                """
                Declaring this class should throw an error.
                """
                pass

    def test_overwrite_method_with_superclass_both_wrong(self):
        """Warning should reference the SUPERCLASS."""
        expected_error_message = "The class 'MiddleNumber' has overwritten the following methods: {'add_5'}."

        class MiddleNumber(self.SimpleNumber):
            """An intermediate number."""
            def add_5(self):
                return super().add_5()

        with self.assertWarnsRegex(errors.OverwrittenMethodWarning, expected_error_message):
            @self.SquareResult()
            class NewNumber(MiddleNumber):
                """
                Declaring this class should throw an error.
                """
                def add_5(self):
                    return super().add_5()

    def test_missing_requirements(self):
        """Should throw an error."""
        simple_number_class = self.SimpleNumber
        square_result_class = self.SquareResult

        class RequirementDecorator(Decorator):
            """A decorator with a requirement."""
            def __init__(self, **kwargs):
                super().__init__(framework_class=simple_number_class,
                                 required_decorators={square_result_class}, **kwargs)

        with self.assertRaises(errors.MissingRequirementsError):

            @RequirementDecorator()
            class NewNumber(self.SimpleNumber):
                """
                Declaring this class should throw an error.
                """
                pass

    def test_passed_requirements(self):
        """Should not throw an error."""
        simple_number_class = self.SimpleNumber
        square_result_class = self.SquareResult

        class RequirementDecorator(Decorator):
            """A decorator with a requirement."""
            def __init__(self, **kwargs):
                super().__init__(framework_class=simple_number_class,
                                 required_decorators={square_result_class}, **kwargs)

        try:
            @RequirementDecorator(ignore_methods=("__init__", "add_5"))
            @self.SquareResult()
            class NewNumber(self.SimpleNumber):
                """
                Declaring this class should throw an error.
                """
                pass
        except errors.MissingRequirementsError as error:
            self.fail(f"Should not have thrown {str(error)}")


class SignatureTests(unittest.TestCase):
    """
    Test that the signatures have been properly updated.
    """
    def setUp(self):
        """Code to run before each test."""
        class SimpleNumber:
            """
            An example class for a real number.
            """
            __decorators__ = []

            def __init__(self, number):
                self.number = number

            def add_5(self):
                """Add 5 to a number."""
                return self.number + 5

        class SquareResult(Decorator):
            """
            Square the result of a SimpleNumber class.
            """
            def __init__(self, **kwargs):
                super().__init__(framework_class=SimpleNumber, required_decorators={}, **kwargs)

            def _decorate_class(self, cls):
                @wraps_class(cls)
                class InnerClass(cls):
                    """
                    A wrapper for normalising y inputs and variance.
                    """
                    def __init__(self, *args, **kwargs):
                        """Inner initialisation."""
                        super().__init__(*args, **kwargs)

                    def add_5(self):
                        """Square the result of this method."""
                        result = super().add_5()
                        return result ** 2

                return InnerClass

        class SimilarNumber(SimpleNumber):
            """
            A superfluous subclass.
            """
            pass

        self.SimilarNumberBefore = SimilarNumber
        self.SimilarNumberAfter = SquareResult()(self.SimilarNumberBefore)

    def test_signature_before(self):
        """Signature should contain number."""
        self.assertEqual("(self, number)", str(inspect.signature(self.SimilarNumberBefore.__init__)))

    def test_signature_after(self):
        """Signature should contain number."""
        self.assertEqual("(self, number)", str(inspect.signature(self.SimilarNumberAfter.__init__)))

    def test_argspec_before(self):
        """Signature should contain number."""
        processed_args = process_args(self.SimilarNumberBefore.__init__, None, 1)
        self.assertDictEqual({"self": None, "number": 1}, processed_args)

    def test_argspec_after(self):
        """Signature should contain number."""
        processed_args = process_args(self.SimilarNumberAfter.__init__, None, 1)
        self.assertDictEqual({"self": None, "number": 1}, processed_args)
