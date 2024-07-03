"""
Test that the posterior predictions of CWGP models are sensible in various ways.
"""

import numpy as np
import torch
from linear_operator.utils.errors import NanError
from sklearn.preprocessing import StandardScaler

from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.warps import SetWarp, WarpFunction, warpfunctions

from ...cases import VanguardTestCase


class CompositionTests(VanguardTestCase):
    """
    Tests for the composition of warp functions.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.affine = warpfunctions.AffineWarpFunction(1, 2)
        self.sinh = warpfunctions.SinhWarpFunction()

    def test_components(self) -> None:
        """Check the components property."""
        box_cox = warpfunctions.BoxCoxWarpFunction(3)
        composed = self.affine @ self.sinh @ box_cox
        self.assertListEqual([self.affine, self.sinh, box_cox], composed.components)

    def test_bad_compose(self) -> None:
        """Should throw a TypeError."""
        with self.assertRaises(TypeError):
            self.affine.compose("bad")

    def test_warp_compose_with_function(self) -> None:
        """Should still be a warp function."""
        composed = self.affine.compose(self.sinh)
        self.assertIsInstance(composed, WarpFunction)

    def test_matmul_with_function(self) -> None:
        """Should still be a warp function."""
        composed = self.affine @ self.sinh
        self.assertIsInstance(composed, WarpFunction)

    def test_matmul_with_int(self) -> None:
        """Should still be a warp function."""
        composed = self.affine @ 3
        self.assertIsInstance(composed, WarpFunction)

    def test_matmul_with_negative_int(self) -> None:
        """Should raise a TypeError."""
        with self.assertRaises(TypeError):
            _ = self.affine @ -3

    def test_matmul_with_float(self) -> None:
        """Should raise a TypeError."""
        with self.assertRaises(TypeError):
            _ = self.affine @ 2.3

    def test_matmul_with_zero(self) -> None:
        """Should be the identity."""
        composed = self.affine @ 0
        self.assertIsInstance(composed, WarpFunction)
        self.assertEqual("_IdentityWarpFunction", type(composed).__name__)


class AssociativityTests(VanguardTestCase):
    """
    Tests for the associativity of warp functions.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        affine = warpfunctions.AffineWarpFunction(1, 2)
        sinh = warpfunctions.SinhWarpFunction()
        box_cox = warpfunctions.BoxCoxWarpFunction(3)

        self.warp_1 = sinh @ (affine @ box_cox)
        self.warp_2 = (sinh @ affine) @ box_cox
        self.warp_3 = sinh @ affine @ box_cox

        self.x = torch.as_tensor(np.array([1, 2, 3, 4]), dtype=torch.float32)
        self.y = torch.as_tensor(np.array([10, 14, 18, 2]), dtype=torch.float32)

    def test_forward(self) -> None:
        """Results should be the same."""
        np.testing.assert_array_equal(
            self.warp_1(self.x).detach().cpu().numpy(), self.warp_2(self.x).detach().cpu().numpy()
        )
        np.testing.assert_array_equal(
            self.warp_1(self.x).detach().cpu().numpy(), self.warp_3(self.x).detach().cpu().numpy()
        )

    def test_inverse(self) -> None:
        """Results should be the same."""
        np.testing.assert_array_equal(
            self.warp_1.inverse(self.y).detach().cpu().numpy(), self.warp_2.inverse(self.y).detach().cpu().numpy()
        )
        np.testing.assert_array_equal(
            self.warp_1.inverse(self.y).detach().cpu().numpy(), self.warp_3.inverse(self.y).detach().cpu().numpy()
        )

    def test_deriv(self) -> None:
        """Results should be the same."""
        np.testing.assert_array_equal(
            self.warp_1.deriv(self.x).detach().cpu().numpy(), self.warp_2.deriv(self.x).detach().cpu().numpy()
        )
        np.testing.assert_array_equal(
            self.warp_1.deriv(self.x).detach().cpu().numpy(), self.warp_3.deriv(self.x).detach().cpu().numpy()
        )


class ParameterTests(VanguardTestCase):
    DATASET = SyntheticDataset(rng=np.random.default_rng(1234))

    def setUp(self):
        self.rng = np.random.default_rng(1234)

    def test_simple_warp_functions_are_different(self) -> None:
        """Two distinct controller instances should have different warp function."""
        affine = warpfunctions.AffineWarpFunction(a=1, b=2)

        @SetWarp(affine, ignore_methods=("__init__",))
        class TestController(GaussianGPController):
            """A test controller."""

        scaler = StandardScaler()
        gp_1 = TestController(
            scaler.fit_transform(self.DATASET.train_x),
            self.DATASET.train_y,
            ScaledRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=self.rng,
        )

        gp_2 = TestController(
            scaler.fit_transform(self.DATASET.train_x),
            self.DATASET.train_y,
            ScaledRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=self.rng,
        )

        self.assertIsNot(gp_1.warp, gp_2.warp)
        self.assertIsNot(gp_1.warp.a, gp_2.warp.a)
        self.assertIsNot(gp_1.warp.b, gp_2.warp.b)

    def test_complicated_warp_functions_are_different(self) -> None:
        """Two distinct, composed controller instances should have different warp function."""
        affine_1 = warpfunctions.AffineWarpFunction(a=1, b=2)
        sinh = warpfunctions.SinhWarpFunction()
        box_cox = warpfunctions.BoxCoxWarpFunction(3)
        affine_2 = warpfunctions.PositiveAffineWarpFunction(a=1, b=2)

        @SetWarp(affine_1 @ sinh @ box_cox @ affine_2, ignore_methods=("__init__",))
        class TestController(GaussianGPController):
            """A test controller."""

        scaler = StandardScaler()
        gp_1 = TestController(
            scaler.fit_transform(self.DATASET.train_x),
            self.DATASET.train_y,
            ScaledRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=self.rng,
        )

        gp_2 = TestController(
            scaler.fit_transform(self.DATASET.train_x),
            self.DATASET.train_y,
            ScaledRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=self.rng,
        )

        for component_1, component_2 in zip(gp_1.warp.components, gp_2.warp.components):
            self.assertIsNot(component_1, component_2)
            for param_1, param_2 in zip(component_1.parameters(), component_2.parameters()):
                self.assertIsNot(param_1, param_2)

    def test_repeated_warp_functions_are_different(self) -> None:
        """The components of a repeated warp function should be different."""
        affine = warpfunctions.AffineWarpFunction(a=1, b=2)

        @SetWarp(affine @ 2, ignore_methods=("__init__",))
        class TestController(GaussianGPController):
            """A test controller."""

        scaler = StandardScaler()
        gp = TestController(
            scaler.fit_transform(self.DATASET.train_x),
            self.DATASET.train_y,
            ScaledRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=self.rng,
        )

        affine_1, affine_2 = gp.warp.components
        self.assertIsNot(affine_1, affine_2)
        for param_1, param_2 in zip(affine_1.parameters(), affine_2.parameters()):
            self.assertIsNot(param_1, param_2)

    def test_frozen_warp_parameters_do_not_change(self) -> None:
        """Parameters of a frozen warp should not be altered by fitting."""
        affine = warpfunctions.AffineWarpFunction(a=1, b=2).freeze()
        box_cox = warpfunctions.SinhWarpFunction()

        @SetWarp(affine @ box_cox, ignore_methods=("__init__",))
        class TestController(GaussianGPController):
            """A test controller."""

        scaler = StandardScaler()
        gp = TestController(
            scaler.fit_transform(self.DATASET.train_x),
            self.DATASET.train_y,
            ScaledRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=self.rng,
        )

        gp.fit(100)
        fitted_affine, _ = gp.warp.components
        self.assertEqual(1, fitted_affine.a.item())
        self.assertEqual(2, fitted_affine.b.item())

    def test_simple_warp_parameters_do_change(self) -> None:
        """Parameters should change after fitting."""
        affine = warpfunctions.AffineWarpFunction(a=1, b=2)

        @SetWarp(affine, ignore_methods=("__init__",))
        class TestController(GaussianGPController):
            """A test controller."""

        scaler = StandardScaler()
        gp = TestController(
            scaler.fit_transform(self.DATASET.train_x),
            self.DATASET.train_y,
            ScaledRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=self.rng,
        )

        gp.fit(100)
        self.assertNotEqual(1, gp.warp.a.item())
        self.assertNotEqual(0, gp.warp.b.item())

    def test_complicated_warp_parameters_do_change(self) -> None:
        """Parameters should change after fitting."""
        arcsinh = warpfunctions.ArcSinhWarpFunction()
        affine_1 = warpfunctions.AffineWarpFunction(a=1, b=2)
        sinh = warpfunctions.SinhWarpFunction()
        affine_2 = warpfunctions.PositiveAffineWarpFunction(a=3, b=10)

        @SetWarp(affine_2 @ sinh @ affine_1 @ arcsinh, ignore_methods=("__init__",))
        class TestController(GaussianGPController):
            """A test controller."""

        scaler = StandardScaler()
        gp = TestController(
            scaler.fit_transform(self.DATASET.train_x),
            self.DATASET.train_y,
            ScaledRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=self.rng,
        )

        gp.fit(100)
        fitted_affine_2, _, fitted_affine_1, _ = gp.warp.components
        self.assertNotEqual(1, fitted_affine_1.a.item())
        self.assertNotEqual(2, fitted_affine_1.b.item())
        self.assertNotEqual(3, fitted_affine_2.a.item())
        self.assertNotEqual(-1, fitted_affine_2.b.item())

    def test_repeated_warp_parameters_do_change(self) -> None:
        """Parameters should change after fitting."""
        affine = warpfunctions.AffineWarpFunction(a=1, b=2)

        @SetWarp(affine @ 2, ignore_methods=("__init__",))
        class TestController(GaussianGPController):
            """A test controller."""

        scaler = StandardScaler()
        gp = TestController(
            scaler.fit_transform(self.DATASET.train_x),
            self.DATASET.train_y,
            ScaledRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=self.rng,
        )

        gp.fit(100)
        fitted_affine_1, fitted_affine_2 = gp.warp.components
        a_1, b_1 = fitted_affine_1.a.item(), fitted_affine_1.a.item()
        a_2, b_2 = fitted_affine_2.b.item(), fitted_affine_2.b.item()
        self.assertNotEqual(1, a_1)
        self.assertNotEqual(2, b_1)
        self.assertNotEqual(1, a_2)
        self.assertNotEqual(2, b_2)
        self.assertNotEqual(a_1, a_2)
        self.assertNotEqual(b_1, b_2)


class ConstraintTests(VanguardTestCase):
    """
    Test that warp functions can be constrained.
    """

    DATASET = SyntheticDataset(rng=np.random.default_rng(1234))

    def setUp(self):
        self.rng = np.random.default_rng(1234)

    def test_fitting_with_unconstrained_warp(self) -> None:
        """Should throw a RuntimeError."""
        box_cox = warpfunctions.BoxCoxWarpFunction(lambda_=0)
        affine = warpfunctions.AffineWarpFunction()

        @SetWarp(box_cox @ affine, ignore_methods=("__init__",))
        class TestController(GaussianGPController):
            """A test controller."""

        scaler = StandardScaler()
        gp = TestController(
            scaler.fit_transform(self.DATASET.train_x),
            self.DATASET.train_y,
            ScaledRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=self.rng,
        )

        expected_regex = r"cholesky_cpu: \d*? of \d*? elements of the torch\.Size\(\[\d*?, \d*?\]\) tensor are NaN\."
        with self.assertRaisesRegex(NanError, expected_regex):
            gp.fit(100)

    def test_fitting_with_constrained_warp(self) -> None:
        """Should NOT throw a RuntimeError."""
        box_cox = warpfunctions.BoxCoxWarpFunction(lambda_=0)
        affine = warpfunctions.PositiveAffineWarpFunction(a=1, b=2)

        @SetWarp(box_cox @ affine, ignore_methods=("__init__",))
        class TestController(GaussianGPController):
            """A test controller."""

        scaler = StandardScaler()
        gp = TestController(
            scaler.fit_transform(self.DATASET.train_x),
            self.DATASET.train_y,
            ScaledRBFKernel,
            y_std=self.DATASET.train_y_std,
            rng=self.rng,
        )

        try:
            gp.fit(100)
        except NanError as error:
            self.fail(f"Should not have thrown: {error!s}")
