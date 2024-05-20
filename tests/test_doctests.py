"""
Doctests for Vanguard.
"""

import doctest
import importlib
import io
import os
import unittest
from types import ModuleType
from typing import Any, Generator, Optional, Tuple

import vanguard


def yield_all_modules(package: ModuleType) -> Generator[ModuleType, None, None]:
    """
    Yield all modules in a package.

    :param package: The package to search.
    """
    package_path = package.__path__[0]
    for root, _, files in os.walk(package_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file[:-3])

                relative_path = os.path.relpath(file_path, package_path)
                relative_path_components = relative_path.split(os.path.sep)

                import_path = ".".join([package.__name__] + [i for i in relative_path_components if i])
                imported_module = importlib.import_module(import_path, package=package.__package__)
                yield imported_module


class DoctestMetaClass(type):
    """
    A metaclass enabling dynamic tests to be rendered as real tests.

    Each module within Vanguard will lead to the creation of a specific test
    method in the class, which will test the doctests.
    """

    def __new__(mcs, name: str, bases: Optional[Tuple[Any]], namespace: Any):
        cls = super().__new__(mcs, name, bases, namespace)

        cls.names_to_suites = {}

        for module in yield_all_modules(vanguard):
            test_suite = doctest.DocTestSuite(module)
            if test_suite.countTestCases():
                test_name = f"test_doctests_in_{module.__name__}"
                cls.names_to_suites[test_name] = test_suite

        for test_name in cls.names_to_suites:

            def inner_test(self) -> None:
                """
                Should not throw any errors.

                Notebook paths are established through a mapping in order to
                avoid unexpected behaviours which occur when using non-local
                loop variables within a new function.
                """
                suite = self.names_to_suites[self._testMethodName]
                result = self.test_runner.run(suite)
                if result.failures or result.errors:
                    for _, result in result.failures:
                        self.fail(result)
                    for _, result in result.errors:
                        self.error(result)

            inner_test.__name__ = test_name
            inner_test.__qualname__ = ".".join((cls.__qualname__, test_name))
            inner_test.__doc__ = "Should not throw any unexpected errors."
            setattr(cls, test_name, inner_test)

        return cls


class Doctests(unittest.TestCase, metaclass=DoctestMetaClass):
    @classmethod
    def setUpClass(cls) -> None:
        """Code to run before all tests."""
        test_stream = io.StringIO()
        cls.test_runner = unittest.TextTestRunner(stream=test_stream)
