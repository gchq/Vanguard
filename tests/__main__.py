"""
Run the tests from the command line.
"""
import argparse
import importlib.util
import unittest
import warnings
from typing import Any, List

TESTS = {
    "u": {"help": "run unit tests.", "path": "units"},
    "d": {"help": "run doctests.", "path": "test_doctests"},
    "e": {"help": "run example notebooks.", "path": "test_examples"},
}


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests for Vanguard.")

    parser.add_argument("components", nargs="*")

    for test_key, test_metadata in TESTS.items():
        argument_key = "-" + test_key
        parser.add_argument(argument_key, dest="tests", action="append_const", const=test_key,
                            help=test_metadata["help"])

    parser.add_argument("-a", "--all", action="store_true", help="Run all tests.")
    parser.add_argument("--show-warnings", action="store_true", help="Display RuntimeWarnings to the user.")
    parser.add_argument("--verbose", action="store_true", help="Increase the verbosity of the test runner.")
    parser.add_argument("--print-output", action="store_true",
                        help="Print 'True' to stdout if all tests pass, or 'False' otherwise.")

    args = parser.parse_args()
    tests_to_be_run = {"u"} if args.tests is None else set(args.tests)
    if args.all:
        tests_to_be_run.update(TESTS)
    args.tests = tests_to_be_run
    return args


def run_tests() -> None:
    """Run all of the tests."""
    args = parse_args()

    if not args.show_warnings:
        warnings.filterwarnings("ignore", category=Warning)

    all_tests = unittest.TestSuite()

    for test_key in args.tests:
        test_path = TESTS[test_key]["path"]

        full_top_package_path = get_absolute_import_path(test_path)
        spec = importlib.util.find_spec(full_top_package_path)
        try:
            import_path_is_package = spec.origin.endswith("__init__.py")
        except AttributeError:
            print(f"{full_top_package_path} could not be loaded.")
            continue

        if import_path_is_package:
            tests = get_tests_from_package(test_path, args.components)
        else:
            tests = get_tests_from_module(test_path)

        all_tests.addTests(tests)

    verbosity = 2 if args.verbose else 1
    test_runner = unittest.TextTestRunner(verbosity=verbosity)

    result = test_runner.run(all_tests)

    if args.print_output:
        print(result.wasSuccessful())


def get_tests_from_module(relative_module_path: str) -> unittest.TestSuite:
    """Get tests from a module."""
    module = importlib.import_module(get_absolute_import_path(relative_module_path), package=__package__)
    tests = unittest.defaultTestLoader.loadTestsFromModule(module)
    return tests


def get_tests_from_package(relative_package_path: str, components: List[Any]) -> unittest.TestSuite:
    """Get all tests from a package."""
    try:
        *directory_path_components, test_pattern_component = components
    except ValueError:
        test_loader = unittest.TestLoader()
        tests = test_loader.discover(get_absolute_import_path(relative_package_path), pattern="test_*.py")
    else:
        file_pattern = f"test_{test_pattern_component}"
        relative_import_path_components = [relative_package_path] + directory_path_components + [file_pattern]
        full_import_path = get_absolute_import_path(".".join(relative_import_path_components))

        module = importlib.import_module(full_import_path, package=__package__)
        tests = unittest.defaultTestLoader.loadTestsFromModule(module)

    return tests


def get_absolute_import_path(relative_import_path: str) -> str:
    """Get the absolute import path."""
    return __package__ + "." + relative_import_path


if __name__ == "__main__":
    run_tests()
