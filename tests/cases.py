"""
Contains test cases for Vanguard testing.
"""
import unittest

import numpy as np
from scipy import stats


class VanguardTestCase(unittest.TestCase):
    """
    A subclass of TestCase designed to check confidence intervals.
    """
    @staticmethod
    def assertInConfidenceInterval(data, interval, delta=0):
        """
        Assert that data is in a confidence interval.

        :param numpy.ndarray data: The data to be tested.
        :param tuple[numpy.ndarray],list[numpy.ndarray] interval: The two interval bounds in the form (lower, upper).
        :param float,int delta: The proportion of elements which can be outside of the interval.
        """
        lower, upper = interval
        elements_outside_interval = (data < lower) | (upper < data)
        number_of_elements_outside_interval = np.sum(elements_outside_interval)
        proportion_of_elements_outside_interval = number_of_elements_outside_interval / len(data)
        if proportion_of_elements_outside_interval > delta:
            error_message = (f"Elements outside interval: {number_of_elements_outside_interval} / {len(data)} "
                             f"({100 * proportion_of_elements_outside_interval:.2f}%) -- delta = {100 * delta:.2f}%")
            raise AssertionError(error_message) from None

    @staticmethod
    def confidence_interval(mu, sigma, alpha):
        """Create a confidence interval."""
        sig_fac = stats.norm.ppf(1 - alpha / 2)
        std_dev = np.sqrt(np.diag(sigma))

        try:
            upper = mu + std_dev * sig_fac
        except ValueError:
            # Assume due to shape mismatch because mu and sigma are from multitask.
            num_points, num_tasks, *_ = mu.shape
            covars = [sigma[nt * num_points: (nt + 1) * num_points, nt * num_points: (nt + 1) * num_points]
                      for nt in range(num_tasks)]
            std_dev = np.stack([np.sqrt(np.diag(cov)) for cov in covars], -1)
            upper = mu + std_dev * sig_fac
        lower = mu - std_dev * sig_fac
        return lower, upper
