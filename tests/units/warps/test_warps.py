"""
Test that the behaviour of some warp functions.
"""
import unittest
from typing import Union

import numpy as np
import numpy.typing
import torch

from vanguard.warps import MultitaskWarpFunction, WarpFunction, warpfunctions


class AutogradAffineWarpFunction(WarpFunction):
    """
    FOR TESTING PURPOSES ONLY.
    We want to test autograd is working for warp derivs, so this AffineWarp uses the default autograd deriv.
    A warp of form y |-> ay + b.
    """
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1, bias=True)
        self.layer.weight.data.fill_(1.)
        self.layer.bias.data.fill_(0.)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.layer(y.reshape(-1, 1))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.div(x - self.layer.bias, self.layer.weight)


class AutogradBoxCoxWarpFunction(WarpFunction):
    """
    FOR TESTING PURPOSES ONLY.
    We want to test autograd is working for warp derivs, so this BoxCoxWarp uses the default autograd deriv.
    """
    def __init__(self, lambda_: Union[float, int] = 0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.lambda_ == 0:
            return torch.log(y)
        else:
            return (torch.sign(y) * torch.abs(y) ** self.lambda_ - 1) / self.lambda_

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.lambda_ == 0:
            return torch.exp(x)
        else:
            return torch.sign(self.lambda_ * x + 1) * torch.abs(self.lambda_ * x + 1) ** (1 / self.lambda_)


class ForwardTest(unittest.TestCase):
    """
    Test the forward of some warps has correct values.
    """
    def setUp(self) -> None:
        """Code to run before each test."""
        self.x = np.array([1, 2])
        self.y = torch.tensor([np.e, np.e ** 2]).float()

    def test_affine_log_value(self) -> None:
        warp = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        x = warp(self.y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), self.x)

    def test_positive_affine_log_value(self) -> None:
        box_cox = warpfunctions.BoxCoxWarpFunction(lambda_=0)
        affine = warpfunctions.PositiveAffineWarpFunction()
        affine.activate(train_y=self.y.detach().cpu().numpy())
        warp = box_cox @ affine
        x = warp(self.y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), self.x)

    def test_affine_non_trivial_log_value(self) -> None:
        warp = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=2, b=1)
        x = warp(self.y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), np.array([np.log(2 * np.e + 1), np.log(2 * (np.e ** 2) + 1)]))

    def test_affine_log_multitask_value(self) -> None:
        warp1 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        warp2 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e)
        warp3 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e ** 2)
        warp = MultitaskWarpFunction(warp1, warp2, warp3)
        y1, y2, y3 = self.y, self.y * np.e, self.y * np.e ** 2
        x1, x2, x3 = self.x, self.x + 2, self.x + 4
        multi_y = torch.stack([y1, y2, y3]).T
        multi_x = np.stack([x1, x2, x3]).T
        x = warp(multi_y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x, multi_x)

    def test_positive_affine_log_value_with_2_dim_y(self) -> None:
        box_cox = warpfunctions.BoxCoxWarpFunction(lambda_=0)
        affine = warpfunctions.PositiveAffineWarpFunction()
        affine.activate(train_y=self.y.unsqueeze(-1).detach().cpu().numpy())
        warp = box_cox @ affine
        x = warp(self.y.unsqueeze(-1)).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), self.x)


class InverseTest(unittest.TestCase):
    """
    Test the inverse of some warps has correct values.
    """
    def setUp(self) -> None:
        """Code to run before each test."""
        self.x = torch.tensor([1, 2]).float()
        self.y = np.array([np.e, np.e ** 2])

    def test_affine_log_value(self) -> None:
        warp = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        y = warp.inverse(self.x).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(y.ravel(), self.y)

    def test_positive_affine_log_value(self) -> None:
        box_cox = warpfunctions.BoxCoxWarpFunction(lambda_=0)
        affine = warpfunctions.PositiveAffineWarpFunction()
        affine.activate(train_y=self.y)
        warp = box_cox @ affine
        y = warp.inverse(self.x).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(y.ravel(), self.y)

    def test_affine_log_multitask_value(self) -> None:
        warp1 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        warp2 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e)
        warp3 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e ** 2)
        y1, y2, y3 = self.y, self.y * np.e, self.y * np.e ** 2
        x1, x2, x3 = self.x, self.x + 2, self.x + 4
        multi_y = np.stack([y1, y2, y3]).T
        multi_x = torch.stack([x1, x2, x3]).T
        warp = MultitaskWarpFunction(warp1, warp2, warp3)
        y = warp.inverse(multi_x).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(y, multi_y, decimal=4)


class DerivTest(unittest.TestCase):
    """
    Test the deriv of some warps has correct values.
    """
    def setUp(self) -> None:
        """Code to run before each test."""
        self.x = np.array([1/2, 1/3])
        self.y = torch.tensor([2, 3]).float()

    def test_affine_log_value(self) -> None:
        warp = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        x = warp.deriv(self.y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), self.x)

    def test_positive_affine_log_value(self) -> None:
        box_cox = warpfunctions.BoxCoxWarpFunction(lambda_=0)
        affine = warpfunctions.PositiveAffineWarpFunction()
        affine.activate(train_y=self.y.detach().cpu().numpy())
        warp = box_cox @ affine
        x = warp.deriv(self.y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), self.x)

    def test_autograd_affine(self) -> None:
        """Test the autograd method on an affine warp."""
        warp = AutogradAffineWarpFunction()
        x = warp.deriv(self.y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), np.array([1, 1]))

    def test_autograd_log(self) -> None:
        """Test the autograd method on a log warp."""
        warp = AutogradBoxCoxWarpFunction(lambda_=0)
        x = warp.deriv(self.y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), self.x)

    def test_autograd_log_affine_multitask(self) -> None:
        """Test the autograd method on a log multitask warp."""
        warp1 = AutogradBoxCoxWarpFunction(lambda_=0)
        warp2 = AutogradAffineWarpFunction()
        warp = MultitaskWarpFunction(warp1, warp2)
        y1, y2 = self.y / 10, self.y + 3
        x1, x2 = 10 * self.x, np.array([1, 1])
        multi_y = torch.stack([y1, y2]).T
        multi_x = np.stack([x1, x2]).T
        x = warp.deriv(multi_y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x, multi_x)


class PositiveAffineWarpTests(unittest.TestCase):
    """
    Tests for the PositiveAffineWarpFunction class.
    """
    NUM_TRAINING_POINTS = 20
    NUM_TESTING_POINTS = 1_000
    TRAINING_POINT_RANGE = 20
    TESTING_POINT_RANGE = 100

    def setUp(self) -> None:
        """Code to run before each test."""
        self.train_x = (np.random.random(self.NUM_TRAINING_POINTS) * self.TRAINING_POINT_RANGE
                        - self.TRAINING_POINT_RANGE / 2)

        self.test_a_b_points = (np.random.random((self.NUM_TESTING_POINTS, 2)) * self.TESTING_POINT_RANGE
                                - self.TESTING_POINT_RANGE / 2)

    def test_feasible_region(self) -> None:
        """Should return correct feasible region."""
        train_x = (np.random.random(self.NUM_TRAINING_POINTS) * self.TRAINING_POINT_RANGE
                   - self.TRAINING_POINT_RANGE / 2)
        all_positive_train_x = np.abs(train_x)
        all_negative_train_x = -all_positive_train_x

        for x_values, indicator in zip((train_x, all_positive_train_x, all_negative_train_x), ("+-", "+", "-")):
            with self.subTest(plus_or_minus=indicator):
                in_boundary_for_all_points = self._get_points_in_boundary_of_feasible_region(self.test_a_b_points,
                                                                                             x_values)

                affine_constraint_points = warpfunctions.PositiveAffineWarpFunction._get_constraint_slopes(x_values)
                in_boundary_for_two_points = self._get_points_in_boundary_of_feasible_region(self.test_a_b_points,
                                                                                             affine_constraint_points)
                np.testing.assert_array_equal(in_boundary_for_all_points, in_boundary_for_two_points)

    def test_feasible_region_no_points(self) -> None:
        """Should raise a ValueError."""
        with self.assertRaises(ValueError):
            warpfunctions.PositiveAffineWarpFunction._get_constraint_slopes(np.array([]))

    @staticmethod
    def _get_points_in_boundary_of_feasible_region(a_b_points: numpy.typing.NDArray[np.floating],
                                                   x_values: numpy.typing.NDArray[np.floating]
                                                   ) -> numpy.typing.NDArray[np.bool_]:
        """Return a boolean array denoting which (a, b) points satisfy ax_i + b for x_i in x_values."""
        a_points = np.repeat(a_b_points[:, 0].reshape(1, -1), len(x_values), axis=0)
        b_points = np.repeat(a_b_points[:, 1].reshape(1, -1), len(x_values), axis=0)
        x_points = np.array(x_values).reshape(-1, 1)
        in_boundary = np.product(a_points * x_points + b_points > 0, axis=0, dtype=bool)
        return in_boundary
