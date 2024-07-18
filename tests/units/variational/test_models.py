"""Tests for vanguard.variational.models."""

import pytest

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.variational.models import SVGPModel


class TestSVGPModel:
    def test_multitask_forbidden(self):
        """
        Test that an appropriate error is raised to guard against naively using SVGPModel in a multitask setting.

        Note that this is quite a contrived example, as most methods that would otherwise lead to using SVGPModel in
        a multitask setting (e.g. the `@Multitask` decorator) don't cause this error to be raised; given that in
        test_decorator_combinations, the `@Multitask` + `@VariationalInference` combination seems to work without
        throwing any errors, it seems that this is simply not an error that is likely to occur in practice.
        """

        class MultitaskSVGP(SVGPModel):
            """SVGP model that pretends to be in a multitask setting."""

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
            )
