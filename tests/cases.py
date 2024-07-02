"""
Contains test cases for Vanguard testing.
"""

import contextlib
import unittest
import warnings
from functools import wraps
from typing import Callable, Tuple, Type, TypeVar, Union

import numpy as np
import numpy.typing
from scipy import stats
from typing_extensions import ParamSpec


class VanguardTestCase(unittest.TestCase):
    """
    A subclass of TestCase designed to check confidence intervals.
    """

    @staticmethod
    def assertInConfidenceInterval(  # pylint: disable=invalid-name
        data: numpy.typing.NDArray[np.floating],
        interval: Tuple[numpy.typing.NDArray[np.floating], numpy.typing.NDArray[np.floating]],
        delta: Union[int, float] = 0,
    ) -> None:
        """
        Assert that data is in a confidence interval.

        :param data: The data to be tested.
        :param interval: The two interval bounds in the form (lower, upper).
        :param delta: The proportion of elements which can be outside of the interval.
        """
        lower, upper = interval
        elements_outside_interval = (data < lower) | (upper < data)
        number_of_elements_outside_interval = np.sum(elements_outside_interval)
        proportion_of_elements_outside_interval = number_of_elements_outside_interval / len(data)
        if proportion_of_elements_outside_interval > delta:
            error_message = (
                f"Elements outside interval: {number_of_elements_outside_interval} / {len(data)} "
                f"({100 * proportion_of_elements_outside_interval:.2f}%) -- delta = {100 * delta:.2f}%"
            )
            raise AssertionError(error_message) from None

    @staticmethod
    def confidence_interval(
        mu: Union[float, numpy.typing.NDArray[np.floating]],
        sigma: Union[float, numpy.typing.NDArray[np.floating]],
        alpha: float,
    ) -> Union[Tuple[float, float], Tuple[numpy.typing.NDArray[np.floating], numpy.typing.NDArray[np.floating]]]:
        """Create a confidence interval."""
        sig_fac = stats.norm.ppf(1 - alpha / 2)
        std_dev = np.sqrt(np.diag(sigma))

        try:
            upper = mu + std_dev * sig_fac
        except ValueError:
            # Assume due to shape mismatch because mu and sigma are from multitask.
            num_points, num_tasks, *_ = mu.shape
            covars = [
                sigma[nt * num_points : (nt + 1) * num_points, nt * num_points : (nt + 1) * num_points]
                for nt in range(num_tasks)
            ]
            std_dev = np.stack([np.sqrt(np.diag(cov)) for cov in covars], -1)
            upper = mu + std_dev * sig_fac
        lower = mu - std_dev * sig_fac
        return lower, upper

    # Ignore invalid-name: we're conforming to the unittest name scheme here, so using camelCase
    @contextlib.contextmanager
    def assertNotWarns(self, expected_warning_type: Type[Warning] = Warning) -> None:  # pylint: disable=invalid-name
        """Assert that enclosed code raises no warnings, or no warnings of a given type."""
        with warnings.catch_warnings(record=True) as ws:
            yield

        ws = list(filter(lambda w: issubclass(w.category, expected_warning_type), ws))

        if len(ws) > 0:
            self.fail(f"Expected no warnings, caught {len(ws)}: {[w.message for w in ws]}")


class FlakyTestError(AssertionError):
    """Raised when a flaky test fails repeatedly."""


P = ParamSpec("P")
T = TypeVar("T")


def flaky(test_method: Callable[P, T]) -> Callable[P, T]:
    """
    Mark a test as flaky - flaky tests are rerun up to 5 times, and pass as soon as they pass at least once.
    """
    max_attempts = 5  # TODO: make this a parameter
    # https://github.com/gchq/Vanguard/issues/195

    @wraps(test_method)
    def repeated_test(self: unittest.TestCase, *args: P.args, **kwargs: P.kwargs) -> T:
        last_attempt = max_attempts - 1
        for attempt_number in range(max_attempts):
            if attempt_number > 0:
                # skip the first setUp as unittest does it for us
                self.setUp()

            try:
                return test_method(self, *args, **kwargs)
            except AssertionError as ex:
                if attempt_number == last_attempt:
                    raise FlakyTestError(
                        f"Flaky test failed {max_attempts} separate times. Last failure is given above."
                    ) from ex

            if attempt_number != last_attempt:
                # skip the last tearDown as unittest does it for us
                self.tearDown()

    return repeated_test
