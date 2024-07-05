"""
Tests for the LossTracker class.
"""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from math import isnan

from tests.cases import get_default_rng
from vanguard.base import GPController
from vanguard.base.metrics import MetricsTracker, loss
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import PeriodicRBFKernel
from vanguard.vanilla import GaussianGPController


def loss_squared(loss_value: float, controller: GPController) -> float:
    """Calculate the square of the loss for test purposes."""
    return loss(loss_value, controller) ** 2


class BasicTests(unittest.TestCase):
    """
    Basic tests for the loss tracker.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.tracker = MetricsTracker(loss)
        for loss_value in range(100):
            self.tracker.run_metrics(loss_value=loss_value, controller=None)

    def test_get_item(self) -> None:
        """Items should be correct."""
        self.assertDictEqual({"loss": 42}, self.tracker[42])

    def test_negative_item(self) -> None:
        """Should still be correct."""
        self.assertDictEqual({"loss": 99}, self.tracker[-1])

    def test_slice(self) -> None:
        """Should be correct."""
        self.assertDictEqual({"loss": list(range(20, 30))}, self.tracker[20:30])

    def test_out_of_range(self) -> None:
        """Should raise an IndexError."""
        with self.assertRaises(IndexError):
            _ = self.tracker[500]

    def test_bad_type(self) -> None:
        """Should raise a TypeError."""
        with self.assertRaises(TypeError):
            _ = self.tracker["hello"]


class NanTests(unittest.TestCase):
    """
    Tests for late addition of a metric.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.tracker = MetricsTracker()
        for loss_value in range(50):
            self.tracker.run_metrics(loss_value=loss_value, controller=None)

        self.tracker.add_metrics(loss)
        for loss_value in range(50, 100):
            self.tracker.run_metrics(loss_value=loss_value, controller=None)

    def test_get_item_before_50(self) -> None:
        """Items should be correct."""
        self.assertTrue(isnan(self.tracker[42]["loss"]))

    def test_get_item_after_50(self) -> None:
        """Items should be correct."""
        self.assertEqual(67, self.tracker[67]["loss"])

    def test_slice(self) -> None:
        """Should be correct."""
        values = self.tracker[40:60]["loss"]
        self.assertTrue(all(isnan(value) for value in values[:10]))
        self.assertEqual(list(range(50, 60)), values[10:])


class MultipleMetricTests(unittest.TestCase):
    """
    Tests for multiple metrics in the tracker.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.tracker = MetricsTracker(loss, loss_squared)
        for loss_value in range(100):
            self.tracker.run_metrics(loss_value=loss_value, controller=None)

    def test_get_item_before_50(self) -> None:
        """Items should be correct."""
        self.assertDictEqual({"loss": 42, "loss_squared": 42**2}, self.tracker[42])

    def test_negative_item(self) -> None:
        """Should still be correct."""
        self.assertDictEqual({"loss": 99, "loss_squared": 99**2}, self.tracker[-1])

    def test_slice(self) -> None:
        """Should be correct."""
        values = list(range(20, 30))
        self.assertDictEqual({"loss": values, "loss_squared": [value**2 for value in values]}, self.tracker[20:30])


class PrintingTests(unittest.TestCase):
    """
    Tests for printing the progress of the tracker.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.rng = get_default_rng()
        dataset = SyntheticDataset(rng=self.rng)

        self.controller = GaussianGPController(
            train_x=dataset.train_x,
            train_y=dataset.train_y,
            kernel_class=PeriodicRBFKernel,
            y_std=dataset.train_y_std,
            rng=self.rng,
        )

        self.new_stdout = StringIO()

    def test_without_metrics(self) -> None:
        """Should not print anything."""
        with redirect_stdout(self.new_stdout):
            self.controller.fit(2)

        output = self.new_stdout.getvalue()
        try:
            self.assertEqual(0, len(output.splitlines()))
        except AssertionError:
            print(output)
            raise

    def test_with_metrics(self) -> None:
        """Should print ten lines"""
        with redirect_stdout(self.new_stdout):
            with self.controller.metrics_tracker.print_metrics():
                self.controller.fit(10)
        output = self.new_stdout.getvalue()
        try:
            self.assertEqual(10, len(output.splitlines()))
        except AssertionError:
            print(output)
            raise

    def test_metrics_output(self) -> None:
        """Should match the string."""
        with redirect_stdout(self.new_stdout):
            with self.controller.metrics_tracker.print_metrics():
                self.controller.fit(2)

        output = self.new_stdout.getvalue()
        self.assertRegex(output, r"^iteration: 1, loss: \d+\.\d+\niteration: 2, loss: \d+\.\d+")

    def test_metrics_output_with_new_format_string(self) -> None:
        """Should match the string."""
        format_string = "This is a test: {iteration} - {loss}"
        with redirect_stdout(self.new_stdout):
            with self.controller.metrics_tracker.print_metrics(format_string=format_string):
                self.controller.fit(1)

        output = self.new_stdout.getvalue()
        self.assertRegex(output, r"^This is a test: \d+ - \d+\.\d")

    def test_metrics_output_with_bad_format_string(self) -> None:
        """Should return a different string."""
        format_string = "{missing}"
        with redirect_stdout(self.new_stdout):
            with self.controller.metrics_tracker.print_metrics(format_string=format_string):
                self.controller.fit(1)

        output = self.new_stdout.getvalue()
        self.assertRegex(output, r"^\d+\.\d+ \(Could not find values for 'missing'\)")

    def test_with_metrics_every(self) -> None:
        """Should print three lines"""
        with redirect_stdout(self.new_stdout):
            with self.controller.metrics_tracker.print_metrics(every=3):
                self.controller.fit(10)
        output = self.new_stdout.getvalue()
        try:
            self.assertEqual(3, len(output.splitlines()))
        except AssertionError:
            print(output)
            raise
