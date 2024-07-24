"""Tests for vanguard.variational.models."""

import pytest
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.variational.models import SVGPModel


class TestSVGPModel:
    def test_multitask_forbidden_indirect(self):
        """
        Test that an appropriate error is raised to guard against naively using `SVGPModel` in a multitask setting.

        Unlike in `test_multitask_forbidden_direct`, we don't create the `SVGPModel` directly, and instead let the GP
        controller create it for us.

        Note that this is quite a contrived example, as most methods that would otherwise lead to using `SVGPModel` in
        a multitask setting (e.g. the `@Multitask` decorator) don't cause this error to be raised; given that in
        test_decorator_combinations, the `@Multitask` + `@VariationalInference` combination seems to apply without
        throwing any errors (it does throw an error on fitting though), it seems that this is simply not an error
        that is likely to occur in practice.
        """

        class MultitaskSVGP(SVGPModel):
            """SVGP model naively configured for a multitask setting."""

            num_tasks = 4

        class Controller(GaussianGPController):
            """Multitask controller for testing."""

            gp_model_class = MultitaskSVGP

        dataset = SyntheticDataset(rng=get_default_rng())

        with pytest.raises(
            TypeError,
            match=f"You are using a {SVGPModel.__name__} in a multi-task problem. {SVGPModel.__name__} does"
            f"not have the correct variational strategy for multi-task.",
        ):
            Controller(
                train_x=dataset.train_x,
                train_y=dataset.train_y,
                y_std=dataset.train_y_std,
                kernel_class=ScaledRBFKernel,
                gp_kwargs={"n_inducing_points": 5},
                rng=get_default_rng(),
            )

    def test_multitask_forbidden_direct(self) -> None:
        """
        Test that an appropriate error is raised to guard against naively using `SVGPModel` in a multitask setting.

        Unlike `test_multitask_forbidden_indirect`, in this test we simply create the `SVGPModel` subclass directly,
        rather than relying on the GP controller to do it for us.
        """
        rng = get_default_rng()
        dataset = SyntheticDataset(rng=rng, n_train_points=20)

        class MultitaskSVGPModel(SVGPModel):
            """SVGP model naively configured for a multitask setting."""

            num_tasks = 4

        with pytest.raises(
            TypeError,
            match=f"You are using a {SVGPModel.__name__} in a multi-task problem. {SVGPModel.__name__} does"
            f"not have the correct variational strategy for multi-task.",
        ):
            MultitaskSVGPModel(
                train_x=dataset.train_x,
                train_y=dataset.train_y,
                likelihood=GaussianLikelihood(),
                mean_module=ConstantMean(),
                covar_module=ScaledRBFKernel(),
                n_inducing_points=10,
                rng=rng,
            )

    def test_forward(self) -> None:
        """
        Test that the `forward()` method operates as expected using the model.

        We expect that the `forward()` method just uses the mean and covariance modules given.
        """
        # Setup for the model
        example_x = torch.tensor([0.0, 1.0, 2.0])
        example_y = torch.tensor([5.0, 6.0, 7.0])

        # Create the model
        kernel = ScaledRBFKernel()
        model = SVGPModel(
            train_x=example_x,
            train_y=example_y,
            likelihood=GaussianLikelihood(),
            mean_module=ConstantMean(),
            covar_module=kernel,
            n_inducing_points=10,
            rng=get_default_rng(),
        )

        # Set expected outputs - the constant mean used should result in a mean vector
        # of zeros
        expected_mean = torch.tensor([0.0, 0.0, 0.0])

        # Set expected outputs - the covariance matrix should be the kernel evaluated on
        # the provided data
        expected_covariance = kernel(example_x, torch.transpose(example_x, 0, 0)).to_dense()

        # Compute a forward pass on the data
        result = model.forward(torch.tensor([0.0, 1.0, 2.0]))

        # Check outputs match expected
        torch.testing.assert_close(expected_mean, result.loc)
        torch.testing.assert_close(expected_covariance, result.covariance_matrix)
