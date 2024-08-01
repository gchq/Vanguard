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
Contains test cases for Vanguard testing.
"""

import contextlib
import unittest
import warnings
from typing import Tuple, Type, Union

import numpy as np
import numpy.typing
from scipy import stats

DEFAULT_RNG_SEED = 1234


def get_default_rng() -> np.random.Generator:
    """
    Get a random number generator with a default seed.

    Call this function rather than `np.random.default_rng()` to create RNGs for testing. Having a centralised
    function for this means that (a) changing the default seed is easy, and (b) we're able to override it to be
    unseeded to evaluate our tests' sensitivity to the random seed.

    If the default seed doesn't work, use `get_default_rng_override_seed` instead, and check that your test would be
    expected to be sensitive to random seeding. For example, we would expect an optimisation problem to be sensitive
    to random seeding (e.g. some of the tests under `classification` are very sensitive), but we would not expect a
    simpler test to be sensitive, and if it were it may indicate a bug.

    :return: A random number generator.
    """
    # TODO: Implement ability to override this (maybe with an environment variable or Pytest flag?) to allow us to
    #  evaluate sensitivity to random seeds.
    # https://github.com/gchq/Vanguard/issues/300
    return get_default_rng_override_seed(DEFAULT_RNG_SEED)


def get_default_rng_override_seed(seed: int) -> np.random.Generator:
    """
    Get a random number generator with a given seed.

    Call this function rather than `np.random.default_rng()` to create RNGs for testing, but **only if the default seed
    provided by `get_default_rng()` doesn't work**. Having a centralised function for this means that we're able to
    override it to be unseeded to evaluate our tests' sensitivity to the random seed.

    :return: A random number generator.
    """
    return np.random.default_rng(seed)


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
