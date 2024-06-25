"""Tests for utilities shared across the test suite."""

import warnings

from tests.cases import VanguardTestCase


class OtherWarningSubclass(Warning):
    """Warning subclass used for testing."""


class SubclassOfUserWarning(UserWarning):
    """Warning subclass used for testing."""


class TestNotWarns(VanguardTestCase):
    def test_with_no_warnings(self):
        with self.assertNotWarns():
            pass

    def test_fails_with_warning(self):
        with self.assertRaises(self.failureException):
            with self.assertNotWarns():
                warnings.warn("A warning!", UserWarning)

    def test_passes_with_warning_not_of_given_type(self):
        with self.assertNotWarns(UserWarning):
            warnings.warn("A warning!", OtherWarningSubclass)

    def test_fails_with_warning_of_given_type(self):
        with self.assertRaises(self.failureException):
            with self.assertNotWarns(OtherWarningSubclass):
                warnings.warn("A warning!", OtherWarningSubclass)

    def test_fails_with_warning_of_descendant_of_given_type(self):
        with self.assertRaises(self.failureException):
            with self.assertNotWarns(UserWarning):
                warnings.warn("A warning!", SubclassOfUserWarning)
