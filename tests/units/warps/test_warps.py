"""
Test that the behaviour of some warp functions.
"""

import unittest
from typing import Union

import numpy as np
import numpy.typing
import torch

from vanguard.warps import MultitaskWarpFunction, WarpFunction, warpfunctions
from vanguard.warps.basefunction import _IdentityWarpFunction


class AutogradAffineWarpFunction(WarpFunction):
    """
    FOR TESTING PURPOSES ONLY.

    We want to test autograd is working for warp derivatives, so this AffineWarp uses the default
    autograd derivative. A warp of form :math:`y |-> ay + b`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1, bias=True)
        self.layer.weight.data.fill_(1.0)
        self.layer.bias.data.fill_(0.0)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.layer(y.reshape(-1, 1))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.div(x - self.layer.bias, self.layer.weight)


class AutogradBoxCoxWarpFunction(WarpFunction):
    """
    FOR TESTING PURPOSES ONLY.

    We want to test autograd is working for warp derivatives, so this BoxCoxWarp uses the
    default autograd deriv.
    """

    def __init__(self, lambda_: Union[float, int] = 0) -> None:
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
        """Define variables shared across tests."""
        self.x = np.array([1, 2])
        self.y = torch.tensor([np.e, np.e**2]).float()

    def test_affine_log_value(self) -> None:
        """Test BoxCoxWarpFunction composed with AffineWarpFunction with known outputs."""
        warp = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        x = warp(self.y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), self.x)

    def test_positive_affine_invalid_params(self) -> None:
        """Test PositiveAffineWarpFunction with invalid parameter inputs."""
        affine = warpfunctions.PositiveAffineWarpFunction(b=-20.0)
        with self.assertRaises(ValueError):
            affine.activate(train_y=self.y.detach().cpu().numpy())

    def test_positive_affine_log_value(self) -> None:
        """Test BoxCoxWarpFunction composed with PositiveAffineWarpFunction with known outputs."""
        box_cox = warpfunctions.BoxCoxWarpFunction(lambda_=0)
        affine = warpfunctions.PositiveAffineWarpFunction()
        affine.activate(train_y=self.y.detach().cpu().numpy())
        warp = box_cox @ affine
        x = warp(self.y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), self.x)

    def test_affine_non_trivial_log_value(self) -> None:
        """Test BoxCoxWarpFunction composed with AffineWarpFunction with known outputs."""
        warp = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=2, b=1)
        x = warp(self.y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), np.array([np.log(2 * np.e + 1), np.log(2 * (np.e**2) + 1)]))

    def test_positive_affine_log_value_with_2_dim_y(self) -> None:
        """Test BoxCoxWarpFunction composed with AffineWarpFunction in multiple dimensions."""
        box_cox = warpfunctions.BoxCoxWarpFunction(lambda_=0)
        affine = warpfunctions.PositiveAffineWarpFunction()
        affine.activate(train_y=self.y.unsqueeze(-1).detach().cpu().numpy())
        warp = box_cox @ affine
        x = warp(self.y.unsqueeze(-1)).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), self.x)

    def test_affine_log_multitask_value(self) -> None:
        """Test BoxCoxWarpFunction composed with many AffineWarpFunctions with known outputs."""
        warp1 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        warp2 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e)
        warp3 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e**2)
        warp = MultitaskWarpFunction(warp1, warp2, warp3)
        y1, y2, y3 = self.y, self.y * np.e, self.y * np.e**2
        x1, x2, x3 = self.x, self.x + 2, self.x + 4
        multi_y = torch.stack([y1, y2, y3]).T
        multi_x = np.stack([x1, x2, x3]).T
        x = warp(multi_y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x, multi_x)

    def test_num_tasks(self) -> None:
        """Test that the number of tasks is correctly defined in a multitask warping."""
        warp1 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        warp2 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e)
        warp = MultitaskWarpFunction(warp1, warp2)
        self.assertEqual(warp.num_tasks, 2)

    def test_compose_multitask(self) -> None:
        """Test composition of multitask warp functions."""
        warp1 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        warp2 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e)
        warp3 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e**2)
        warp = MultitaskWarpFunction(warp1, warp2) @ MultitaskWarpFunction(warp2, warp3)

        y1, y2, y3 = self.y, self.y * np.e, self.y * np.e**2
        multi_y = torch.stack([y1, y2, y3]).T
        result = warp(multi_y).detach().cpu().numpy()

        # Check the result is not nan
        self.assertFalse(np.isnan(result).any())

    def test_compose_multitask_invalid(self) -> None:
        """Test invalid composition of multitask warp functions."""
        warp1 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        warp2 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e)
        warp3 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e**2)

        # We should not be able to compose multitask and non-multitask warp functions
        with self.assertRaises(TypeError):
            # pylint: disable=expression-not-assigned
            MultitaskWarpFunction(warp1, warp2) @ warp3

    def test_compose_multitask_with_self(self) -> None:
        """Test composition of a multitask warp with itself."""
        warp1 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        warp2 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e)
        warp = MultitaskWarpFunction(warp1, warp2)

        y1, y2, y3 = self.y, self.y * np.e, self.y * np.e**2
        multi_y = torch.stack([y1, y2, y3]).T
        output_pre_compose = warp(multi_y).detach().cpu().numpy()

        # We should not be able to compose a negative number of times
        with self.assertRaises(ValueError):
            warp.compose_with_self(-10)

        # With a zero valued composition, we should recover the identity warp
        np.testing.assert_array_almost_equal(warp.compose_with_self(0)(torch.tensor([2, 3])), torch.tensor([2, 3]))

        # With more than 0 compositions, we should get distinct components
        two_composed = warp.compose_with_self(2)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(output_pre_compose, two_composed(multi_y).detach().cpu().numpy())

    def test_logit(self) -> None:
        """Test forward method of LogitWarpFunction."""
        warp = warpfunctions.LogitWarpFunction()

        # Manually computed :math:`log(0.25/(1-0.25)) = -1.098612`
        np.testing.assert_array_almost_equal(warp(torch.tensor(0.25)), np.array([-1.098612]))

    def test_soft_plus(self) -> None:
        """Test forward method of SoftPlusWarpFunction."""
        warp = warpfunctions.SoftPlusWarpFunction()

        # Manually computed :math:`log(e^0.25 - 1) = -1.258692`
        np.testing.assert_array_almost_equal(warp(torch.tensor(0.25)), np.array([-1.258692]))


class InverseTest(unittest.TestCase):
    """
    Test the inverse of some warps has correct values.
    """

    def setUp(self) -> None:
        """Define variables shared across tests."""
        self.x = torch.tensor([1, 2]).float()
        self.y = np.array([np.e, np.e**2])

    def test_affine_log_value(self) -> None:
        """Test BoxCoxWarpFunction composed with AffineWarpFunction with known outputs."""
        warp = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        y = warp.inverse(self.x).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(y.ravel(), self.y)

    def test_positive_affine_log_value(self) -> None:
        """Test BoxCoxWarpFunction composed with AffineWarpFunction with known outputs."""
        box_cox = warpfunctions.BoxCoxWarpFunction(lambda_=0)
        affine = warpfunctions.PositiveAffineWarpFunction()
        affine.activate(train_y=self.y)
        warp = box_cox @ affine
        y = warp.inverse(self.x).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(y.ravel(), self.y)

    def test_affine_log_multitask_value(self) -> None:
        """Test BoxCoxWarpFunction composed with AffineWarpFunction with multiple tasks."""
        warp1 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        warp2 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e)
        warp3 = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction(a=np.e**2)
        y1, y2, y3 = self.y, self.y * np.e, self.y * np.e**2
        x1, x2, x3 = self.x, self.x + 2, self.x + 4
        multi_y = np.stack([y1, y2, y3]).T
        multi_x = torch.stack([x1, x2, x3]).T
        warp = MultitaskWarpFunction(warp1, warp2, warp3)
        y = warp.inverse(multi_x).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(y, multi_y, decimal=4)

    def test_sinh(self) -> None:
        """Test the inverse of the ArcSinhWarpFunction."""
        warp = warpfunctions.ArcSinhWarpFunction()

        # :math:`sinh(1.5) = 2.12928` can be verified independently
        np.testing.assert_array_almost_equal(warp.inverse(torch.tensor(1.5)), np.array([2.12928]))

    def test_logit(self) -> None:
        """Test inverse method of LogitWarpFunction."""
        warp = warpfunctions.LogitWarpFunction()

        # Manually computed :math:`sigmoid(0.25) = 0.5621765`
        np.testing.assert_array_almost_equal(warp.inverse(torch.tensor(0.25)), np.array([0.5621765]))

    def test_soft_plus(self) -> None:
        """Test inverse method of SoftPlusWarpFunction."""
        warp = warpfunctions.SoftPlusWarpFunction()

        # Manually computed :math:`log(e^0.25 + 1) = 0.8259394`
        np.testing.assert_array_almost_equal(warp.inverse(torch.tensor(0.25)), np.array([0.8259394]))


class DerivativeTest(unittest.TestCase):
    """
    Test the derivatives of some warps has correct values.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.x = np.array([1 / 2, 1 / 3])
        self.y = torch.tensor([2, 3]).float()

    def test_affine_log_value(self) -> None:
        """Test BoxCoxWarpFunction composed with AffineWarpFunction with known outputs."""
        warp = warpfunctions.BoxCoxWarpFunction(lambda_=0) @ warpfunctions.AffineWarpFunction()
        x = warp.deriv(self.y).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(x.ravel(), self.x)

    def test_positive_affine_log_value(self) -> None:
        """Test BoxCoxWarpFunction composed with AffineWarpFunction with known outputs."""
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

    def test_logit(self) -> None:
        """Test derivative method of LogitWarpFunction."""
        warp = warpfunctions.LogitWarpFunction()

        # Manually computed :math:`(1 - 2 * 0.25) / (0.25 * (1 - 0.25)) = 2.666666666`
        np.testing.assert_array_almost_equal(warp.deriv(torch.tensor(0.25)), np.array([2.666666666]))

    def test_soft_plus(self) -> None:
        """Test derivative method of SoftPlusWarpFunction."""
        warp = warpfunctions.SoftPlusWarpFunction()

        # Manually computed :math:`sigmoid(0.25) = 0.5621765`
        np.testing.assert_array_almost_equal(warp.deriv(torch.tensor(0.25)), np.array([0.5621765]))


class PositiveAffineWarpTests(unittest.TestCase):
    """
    Tests for the PositiveAffineWarpFunction class.
    """

    NUM_TRAINING_POINTS = 20
    NUM_TESTING_POINTS = 1_000
    TRAINING_POINT_RANGE = 20
    TESTING_POINT_RANGE = 100

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.generator = np.random.default_rng(1234)
        self.train_x = (
            self.generator.random(self.NUM_TRAINING_POINTS) * self.TRAINING_POINT_RANGE - self.TRAINING_POINT_RANGE / 2
        )

        self.test_a_b_points = (
            self.generator.random((self.NUM_TESTING_POINTS, 2)) * self.TESTING_POINT_RANGE
            - self.TESTING_POINT_RANGE / 2
        )

    def test_feasible_region(self) -> None:
        """Test computation of the feasible region."""
        train_x = (
            self.generator.random(self.NUM_TRAINING_POINTS) * self.TRAINING_POINT_RANGE - self.TRAINING_POINT_RANGE / 2
        )
        all_positive_train_x = np.abs(train_x)
        all_negative_train_x = -all_positive_train_x

        for x_values, indicator in zip((train_x, all_positive_train_x, all_negative_train_x), ("+-", "+", "-")):
            with self.subTest(plus_or_minus=indicator):
                in_boundary_for_all_points = self._get_points_in_boundary_of_feasible_region(
                    self.test_a_b_points, x_values
                )

                # pylint: disable=protected-access
                affine_constraint_points = warpfunctions.PositiveAffineWarpFunction._get_constraint_slopes(x_values)
                in_boundary_for_two_points = self._get_points_in_boundary_of_feasible_region(
                    self.test_a_b_points, affine_constraint_points
                )
                np.testing.assert_array_equal(in_boundary_for_all_points, in_boundary_for_two_points)

    def test_feasible_region_no_points(self) -> None:
        """Test behaviour when the feasible region has no points in it."""
        with self.assertRaises(ValueError):
            # pylint: disable=protected-access
            warpfunctions.PositiveAffineWarpFunction._get_constraint_slopes(np.array([]))

    @staticmethod
    def _get_points_in_boundary_of_feasible_region(
        a_b_points: numpy.typing.NDArray[np.floating], x_values: numpy.typing.NDArray[np.floating]
    ) -> numpy.typing.NDArray[np.bool_]:
        """
        Identify a boolean array denoting which (a, b) points satisfy ax_i + b for x_i in x_values.

        :param a_b_points: Values of a and b to consider.
        :param x_values: Data-points to pass to the linear warping.
        :return: Boolean array with value 1 where corresponding  entries in `x_values` lie in the feasible region.
        """
        a_points = np.repeat(a_b_points[:, 0].reshape(1, -1), len(x_values), axis=0)
        b_points = np.repeat(a_b_points[:, 1].reshape(1, -1), len(x_values), axis=0)
        x_points = np.array(x_values).reshape(-1, 1)
        in_boundary = numpy.prod(a_points * x_points + b_points > 0, axis=0, dtype=bool)
        return in_boundary


class TestIdentityWarp(unittest.TestCase):
    """
    Tests related to the identity warp function.
    """

    def setUp(self) -> None:
        """
        Define data shared across tests
        """
        self.warp_function = _IdentityWarpFunction()
        self.data_point = torch.tensor([1.0, 2.0, 3.0])

    def test_forward(self) -> None:
        """
        Test forward method of the identity warp function.
        """
        np.testing.assert_array_almost_equal(self.warp_function(self.data_point), self.data_point)

    def test_derivative(self) -> None:
        """
        Test derivative method of the identity warp function.
        """
        np.testing.assert_array_almost_equal(self.warp_function.deriv(self.data_point), torch.ones([3]))

    def test_inverse(self) -> None:
        """
        Test inverse method of the identity warp function.
        """
        np.testing.assert_array_almost_equal(self.warp_function.inverse(self.data_point), self.data_point)
