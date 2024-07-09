"""
Tests for the pairwise combinations of decorators.
"""

import contextlib
import itertools
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar
from unittest.mock import patch

import pytest
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import BernoulliLikelihood, DirichletClassificationLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import VariationalELBO

from tests.cases import get_default_rng
from vanguard.base import GPController
from vanguard.base.posteriors import MonteCarloPosteriorCollection
from vanguard.classification import BinaryClassification, DirichletMulticlassClassification
from vanguard.datasets import Dataset
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.datasets.synthetic import SyntheticDataset, complicated_f, simple_f
from vanguard.decoratorutils import Decorator
from vanguard.decoratorutils.errors import MissingRequirementsError, TopmostDecoratorError
from vanguard.distribute import Distributed
from vanguard.hierarchical import BayesianHyperparameters, VariationalHierarchicalHyperparameters
from vanguard.kernels import ScaledRBFKernel
from vanguard.learning import LearnYNoise
from vanguard.multitask import Multitask
from vanguard.multitask.likelihoods import FixedNoiseMultitaskGaussianLikelihood
from vanguard.normalise import NormaliseY
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference
from vanguard.warps import SetInputWarp, SetWarp, warpfunctions

ControllerT = TypeVar("ControllerT", bound=GPController)


@BayesianHyperparameters()
class TestHierarchicalKernel(RBFKernel):
    """A kernel to test Bayesian hierarchical hyperparameters"""


DECORATORS = {
    BinaryClassification: {
        "decorator": {},
        "controller": {
            "y_std": 0,
            "likelihood_class": BernoulliLikelihood,
            "marginal_log_likelihood_class": VariationalELBO,
        },
    },
    DirichletMulticlassClassification: {
        "decorator": {"num_classes": 4},
        "controller": {
            "likelihood_class": DirichletClassificationLikelihood,
            "likelihood_kwargs": {"learn_additional_noise": True},
        },
        "dataset": MulticlassGaussianClassificationDataset(
            num_train_points=10, num_test_points=4, num_classes=4, rng=get_default_rng()
        ),
    },
    Distributed: {"decorator": {"n_experts": 3, "rng": get_default_rng()}, "controller": {}},
    VariationalHierarchicalHyperparameters: {
        "decorator": {"num_mc_samples": 13},
        "controller": {
            "kernel_class": TestHierarchicalKernel,
        },
    },
    LearnYNoise: {"decorator": {}, "controller": {}},
    NormaliseY: {"decorator": {}, "controller": {}},
    Multitask: {
        "decorator": {"num_tasks": 2},
        "controller": {
            "likelihood_class": FixedNoiseMultitaskGaussianLikelihood,
        },
        "dataset": SyntheticDataset(
            [simple_f, complicated_f], n_train_points=10, n_test_points=1, rng=get_default_rng()
        ),
    },
    SetWarp: {
        "decorator": {"warp_function": warpfunctions.SinhWarpFunction()},
        "controller": {},
    },
    SetInputWarp: {
        "decorator": {"warp_function": warpfunctions.SinhWarpFunction()},
        "controller": {},
    },
    VariationalInference: {
        "decorator": {"n_inducing_points": 5},
        "controller": {
            "likelihood_class": FixedNoiseGaussianLikelihood,
            "marginal_log_likelihood_class": VariationalELBO,
        },
    },
}

COMBINATION_CONTROLLER_KWARGS: Dict[Tuple[Type[Decorator], Type[Decorator]], Dict[str, Any]] = {
    (DirichletMulticlassClassification, VariationalInference): {
        "likelihood_class": DirichletClassificationLikelihood,
    },
    (Multitask, VariationalInference): {
        "likelihood_class": FixedNoiseMultitaskGaussianLikelihood,
    },
}

EXCLUDED_COMBINATIONS = {
    (BinaryClassification, Multitask),  # likelihood contradiction
    (BinaryClassification, VariationalInference),  # model contradiction
    (BinaryClassification, DirichletMulticlassClassification),  # can only do classification once
    (DirichletMulticlassClassification, Multitask),  # multiple datasets
    (DirichletMulticlassClassification, VariationalInference),  # model contradiction
    (Distributed, Multitask),  # cannot aggregate multitask predictions (shape errors)
    (Distributed, VariationalHierarchicalHyperparameters),  # cannot combine with a BCM aggregator
    (
        VariationalHierarchicalHyperparameters,
        DirichletMulticlassClassification,
    ),  # can't aggregate multitask predictions
}

EXPECTED_COMBINATION_CREATE_ERRORS = {
    (NormaliseY, DirichletMulticlassClassification): (
        TypeError,
        "For classification, train_y must be integer-valued. Got dtype=.*",
    ),
}

EXPECTED_COMBINATION_FIT_ERRORS = {
    (VariationalInference, Multitask): (RuntimeError, ".* may not be the correct choice for a variational strategy"),
}


@contextlib.contextmanager
def maybe_throws(category: Optional[Type[Exception]], match: Optional[str] = None) -> Optional[pytest.ExceptionInfo]:
    """
    Do nothing if None is given. Do `pytest.raises()` if an exception type is passed.

    :return: None if no exception type was passed. ExceptionInfo from `pytest.raises()` if an exception type was passed.
    """
    if category is None:
        yield
        return None
    else:
        with pytest.raises(category, match=match) as exc:
            yield
        return exc


def _initialise_decorator_pair(
    upper_decorator_details: Tuple[Decorator, Dict], lower_decorator_details: Tuple[Decorator, Dict]
) -> "Tuple[Callable, Callable, dict[str, Any], Dataset]":
    """Initialise a pair of decorators."""
    upper_decorator, upper_controller_kwargs, upper_dataset = _create_decorator(upper_decorator_details)
    lower_decorator, lower_controller_kwargs, lower_dataset = _create_decorator(lower_decorator_details)

    if upper_dataset and lower_dataset:
        raise RuntimeError(
            f"Cannot combine {type(upper_decorator).__name__} and "
            f"{type(lower_decorator).__name__}: two datasets have been passed."
        )

    dataset = (
        upper_dataset or lower_dataset or SyntheticDataset(n_train_points=20, n_test_points=2, rng=get_default_rng())
    )

    controller_kwargs = {**upper_controller_kwargs, **lower_controller_kwargs}
    return upper_decorator, lower_decorator, controller_kwargs, dataset


def _create_decorator(details: Tuple[Callable, Dict[str, Any]]) -> Tuple[Callable, ControllerT, Optional[Dataset]]:
    """Unpack decorator details."""
    decorator_class, all_decorator_kwargs = details
    decorator = decorator_class(ignore_all=True, **all_decorator_kwargs["decorator"])
    return decorator, all_decorator_kwargs["controller"], all_decorator_kwargs.get("dataset", None)


@pytest.mark.parametrize(
    "upper_details, lower_details",
    [
        pytest.param(
            upper_details,
            lower_details,
            id=f"Upper: {upper_details[0].__name__} - Lower: {lower_details[0].__name__}",
        )
        for upper_details, lower_details in itertools.permutations(DECORATORS.items(), r=2)
        if (upper_details[0], lower_details[0]) not in EXCLUDED_COMBINATIONS
        and (lower_details[0], upper_details[0]) not in EXCLUDED_COMBINATIONS
    ],
)
def test_combinations(upper_details: Tuple[Decorator, Dict], lower_details: Tuple[Decorator, Dict]) -> None:
    """Shouldn't throw any errors."""
    upper_decorator, lower_decorator, controller_kwargs, dataset = _initialise_decorator_pair(
        upper_details, lower_details
    )

    final_kwargs = {
        "train_x": dataset.train_x,
        "train_y": dataset.train_y,
        "y_std": dataset.train_y_std,
        "kernel_class": ScaledRBFKernel,
        "rng": get_default_rng(),
    }

    combination = (type(upper_decorator), type(lower_decorator))
    combination_controller_kwargs = COMBINATION_CONTROLLER_KWARGS.get(combination, {})
    final_kwargs.update(controller_kwargs)
    final_kwargs.update(combination_controller_kwargs)

    try:
        controller_class = upper_decorator(lower_decorator(GaussianGPController))
    except (MissingRequirementsError, TopmostDecoratorError):
        return

    expected_error_class, expected_error_message = EXPECTED_COMBINATION_CREATE_ERRORS.get(combination, (None, None))
    with maybe_throws(expected_error_class, expected_error_message):
        controller = controller_class(**final_kwargs)
    if expected_error_class is not None:
        return

    expected_error_class, expected_error_message = EXPECTED_COMBINATION_FIT_ERRORS.get(combination, (None, None))
    with maybe_throws(expected_error_class, expected_error_message):
        controller.fit(1)
    if expected_error_class is not None:
        return

    if hasattr(controller, "classify_points"):
        pass
        # TODO: check that the classification methods don't throw any unexpected errors
        # controller.classify_points(dataset.test_x)
        # controller.classify_fuzzy_points(dataset.test_x, dataset.test_x_std)
    else:
        # check that the prediction methods don't throw any unexpected errors
        posterior = controller.posterior_over_point(dataset.test_x)
        try:
            posterior.prediction()
        except TypeError as exc:
            if isinstance(upper_decorator, SetWarp) or isinstance(lower_decorator, SetWarp):
                assert str(exc) == "The mean and covariance of a warped GP cannot be computed exactly."
            else:
                raise

        posterior.confidence_interval(dataset.significance)

        # we don't care about any kind of accuracy here, so just pick the minimum number that doesn't
        # cause errors
        with patch.object(MonteCarloPosteriorCollection, "INITIAL_NUMBER_OF_SAMPLES", 4):
            fuzzy_posterior = controller.posterior_over_fuzzy_point(dataset.test_x, dataset.test_x_std)

        try:
            fuzzy_posterior.prediction()
        except TypeError as exc:
            if isinstance(upper_decorator, SetWarp) or isinstance(lower_decorator, SetWarp):
                assert str(exc) == "The mean and covariance of a warped GP cannot be computed exactly."
            else:
                raise

        # again, we don't care about accuracy, so just pick the minimum number that doesn't cause errors
        with patch.object(MonteCarloPosteriorCollection, "_decide_mc_num_samples", lambda *_: 4):
            fuzzy_posterior.confidence_interval(dataset.significance)
