"""
Tests for the pairwise combinations of decorators.
"""

import itertools
from typing import Any, Dict, Optional, Tuple, Type, TypeVar
from unittest.mock import patch

import pytest
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import BernoulliLikelihood, DirichletClassificationLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from typing_extensions import TypedDict

from tests.cases import get_default_rng, maybe_throws, maybe_warns

# not super happy about importing HigherRankKernel/HigherRankMean from another test file - these should probably be
# moved to some more central location
from tests.units.test_features import HigherRankKernel, HigherRankMean
from vanguard.base import GPController
from vanguard.base.posteriors import MonteCarloPosteriorCollection
from vanguard.classification import BinaryClassification, CategoricalClassification, DirichletMulticlassClassification
from vanguard.classification.kernel import DirichletKernelMulticlassClassification
from vanguard.classification.likelihoods import DirichletKernelClassifierLikelihood, GenericExactMarginalLogLikelihood
from vanguard.datasets import Dataset
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.datasets.synthetic import HigherRankSyntheticDataset, SyntheticDataset, complicated_f, simple_f
from vanguard.decoratorutils import Decorator
from vanguard.decoratorutils.errors import BadCombinationWarning, MissingRequirementsError, TopmostDecoratorError
from vanguard.distribute import Distributed
from vanguard.features import HigherRankFeatures
from vanguard.hierarchical import (
    BayesianHyperparameters,
    LaplaceHierarchicalHyperparameters,
    VariationalHierarchicalHyperparameters,
)
from vanguard.kernels import ScaledRBFKernel
from vanguard.learning import LearnYNoise
from vanguard.multitask import Multitask
from vanguard.multitask.likelihoods import FixedNoiseMultitaskGaussianLikelihood
from vanguard.normalise import NormaliseY
from vanguard.standardise import DisableStandardScaling
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference
from vanguard.warps import SetInputWarp, SetWarp, warpfunctions

ControllerT = TypeVar("ControllerT", bound=GPController)


@BayesianHyperparameters()
class TestHierarchicalKernel(RBFKernel):
    """A kernel to test Bayesian hierarchical hyperparameters"""


class DecoratorDetails(TypedDict, total=False):
    """Type hint for the decorator details specification in `DECORATORS`."""

    decorator: Dict[str, Any]
    controller: Dict[str, Any]
    dataset: Dataset


DECORATORS: Dict[Type[Decorator], DecoratorDetails] = {
    BinaryClassification: {
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
            "kernel_class": ScaledRBFKernel,
            "kernel_kwargs": {"batch_shape": (4,)},
        },
        # TODO: fails with a shape mismatch error for any fewer than 40 points
        # https://github.com/gchq/Vanguard/issues/322
        "dataset": MulticlassGaussianClassificationDataset(
            num_train_points=40, num_test_points=4, num_classes=4, rng=get_default_rng()
        ),
    },
    DirichletKernelMulticlassClassification: {
        "decorator": {"num_classes": 4},
        "controller": {
            "mean_class": ZeroMean,
            "kernel_class": RBFKernel,
            "likelihood_class": DirichletKernelClassifierLikelihood,
            "marginal_log_likelihood_class": GenericExactMarginalLogLikelihood,
        },
        "dataset": MulticlassGaussianClassificationDataset(
            num_train_points=20, num_test_points=4, num_classes=4, rng=get_default_rng()
        ),
    },
    HigherRankFeatures: {
        "decorator": {"rank": 2},
        "controller": {
            "kernel_class": HigherRankKernel,
            "mean_class": HigherRankMean,
        },
        "dataset": HigherRankSyntheticDataset(n_train_points=10, n_test_points=4, rng=get_default_rng()),
    },
    DisableStandardScaling: {},
    CategoricalClassification: {
        "decorator": {"num_classes": 4},
        "dataset": MulticlassGaussianClassificationDataset(
            num_train_points=10, num_test_points=4, num_classes=4, rng=get_default_rng()
        ),
    },
    Distributed: {"decorator": {"n_experts": 3, "rng": get_default_rng()}},
    VariationalHierarchicalHyperparameters: {
        "decorator": {"num_mc_samples": 13},
        "controller": {"kernel_class": TestHierarchicalKernel},
    },
    # LaplaceHierarchicalHyperparameters: {
    #     "decorator": {"num_mc_samples": 13},
    #     "controller": {"kernel_class": TestHierarchicalKernel},
    # },
    LearnYNoise: {},
    NormaliseY: {},
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
    },
    SetInputWarp: {
        "decorator": {"warp_function": warpfunctions.SinhWarpFunction()},
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
    # multitask classification generally doesn't work:
    (Multitask, BinaryClassification),  # likelihood contradiction
    (Multitask, CategoricalClassification),  # multiple datasets
    (Multitask, DirichletMulticlassClassification),  # multiple datasets
    (Multitask, DirichletKernelMulticlassClassification),  # multiple datasets
    # nor does classification with variational inference:
    (VariationalInference, BinaryClassification),  # model contradiction
    (VariationalInference, DirichletKernelMulticlassClassification),  # model contradiction
    # conflicts with Distributed:
    (Distributed, Multitask),  # cannot aggregate multitask predictions (shape errors)
    (Distributed, VariationalHierarchicalHyperparameters),  # cannot combine with a BCM aggregator
    # can't aggregate multitask predictions:
    (VariationalHierarchicalHyperparameters, DirichletMulticlassClassification),
    (VariationalHierarchicalHyperparameters, DirichletKernelMulticlassClassification),
    # HigherRankFeatures has dataset conflicts with several other decorators:
    (HigherRankFeatures, DirichletMulticlassClassification),  # two datasets
    (HigherRankFeatures, DirichletKernelMulticlassClassification),  # two datasets
    (HigherRankFeatures, CategoricalClassification),  # two datasets
    (HigherRankFeatures, Multitask),  # two datasets
    # TEMPORARY - TO FIX:
    # TODO(rg): Fails with an "index out of bounds" error - seems to be because the warp function moves the class
    #  indices out of the expected range. The other classification decorators seem to work though!
    # https://github.com/gchq/Vanguard/issues/376
    (DirichletKernelMulticlassClassification, SetWarp),
    # TODO(rg): Fails due to shape mismatch whichever one is on top. When VHR is on top of HRF this makes sense, but
    #  the other way around should probably work. Will require a custom @BayesianHyperparameters higher-rank kernel
    #  class.
    # https://github.com/gchq/Vanguard/issues/375
    (HigherRankFeatures, VariationalHierarchicalHyperparameters),
}

# Errors we expect to be raised on initialisation of the decorated class.
EXPECTED_COMBINATION_INIT_ERRORS: Dict[Tuple[Type[Decorator], Type[Decorator]], Tuple[Type[Exception], str]] = {
    (NormaliseY, DirichletMulticlassClassification): (
        TypeError,
        "NormaliseY should not be used above classification decorators.",
    ),
}

# Errors we expect to be raised on decorator application.
EXPECTED_COMBINATION_APPLY_ERRORS: Dict[Tuple[Type[Decorator], Type[Decorator]], Tuple[Type[Exception], str]] = {
    (Distributed, HigherRankFeatures): (
        TypeError,
        ".* cannot handle higher-rank features. Consider moving the `@Distributed` decorator "
        "below the `@HigherRankFeatures` decorator.",
    ),
    # can only use one hyperparameter decorator at once:
    **{
        (upper, lower): (
            TypeError,
            f"This class is already decorated with `{lower.__name__}`. "
            f"Please use only one hierarchical hyperparameters decorator at once.",
        )
        for upper, lower in itertools.permutations(
            [VariationalHierarchicalHyperparameters, LaplaceHierarchicalHyperparameters], r=2
        )
    },
    # can only use one classification decorator at a time:
    **{
        (upper, lower): (
            TypeError,
            "This class is already decorated with a classification decorator. "
            "Please use only one classification decorator at once.",
        )
        for upper, lower in itertools.permutations(
            [
                # Note that we _don't_ include binary or categorical classification - they will instead raise
                # a `MissingRequirementsError` for missing `VariationalInference`.
                DirichletMulticlassClassification,
                DirichletKernelMulticlassClassification,
            ],
            r=2,
        )
    },
}

# Warnings we expect to be raised on decorator application.
EXPECTED_COMBINATION_APPLY_WARNINGS: Dict[Tuple[Type[Decorator], Type[Decorator]], Tuple[Type[Warning], str]] = {
    (NormaliseY, DirichletMulticlassClassification): (
        BadCombinationWarning,
        "NormaliseY should not be used above classification decorators - this may lead to unexpected behaviour.",
    ),
}

# Errors we expect to be raised when calling controller.fit().
EXPECTED_COMBINATION_FIT_ERRORS: Dict[Tuple[Type[Decorator], Type[Decorator]], Tuple[Type[Exception], str]] = {
    (VariationalInference, Multitask): (RuntimeError, ".* may not be the correct choice for a variational strategy"),
}

# Combinations for which we ignore the normal error raised when we try and pass two datasets. This is only for
# testing errors raised on decorator application - these combinations *must* raise an exception on application,
# as initialisation will always fail with an `AttributeError` since `_initialise_decorator_pair` will return
# `dataset=None`.
DATASET_CONFLICT_OVERRIDES = {
    # Ignore combinations of classification decorators which might have conflicting dataset requirements - applying
    # two classification decorators raises an error which we want to check for.
    *{
        (upper, lower)
        for upper, lower in itertools.permutations(
            [
                CategoricalClassification,
                DirichletMulticlassClassification,
                DirichletKernelMulticlassClassification,
            ],
            r=2,
        )
    }
}


def _initialise_decorator_pair(
    upper_decorator_details: Tuple[Type[Decorator], DecoratorDetails],
    lower_decorator_details: Tuple[Type[Decorator], DecoratorDetails],
) -> Tuple[Decorator, Decorator, Dict[str, Any], Optional[Dataset]]:
    """Initialise a pair of decorators."""
    upper_decorator, upper_controller_kwargs, upper_dataset = _create_decorator(upper_decorator_details)
    lower_decorator, lower_controller_kwargs, lower_dataset = _create_decorator(lower_decorator_details)

    if (type(upper_decorator), type(lower_decorator)) in DATASET_CONFLICT_OVERRIDES:
        # Decorator application *must* fail, so passing dataset=None doesn't matter as we'll never reach initialisation
        dataset = None
    elif upper_dataset and lower_dataset:
        # Passing two datasets is ambiguous!
        raise RuntimeError(
            f"Cannot combine {type(upper_decorator).__name__} and "
            f"{type(lower_decorator).__name__}: two datasets have been passed."
        )
    else:
        # Pass whichever dataset we have, or a default.
        dataset = (
            upper_dataset
            or lower_dataset
            or SyntheticDataset(n_train_points=20, n_test_points=2, rng=get_default_rng())
        )

    controller_kwargs = {**upper_controller_kwargs, **lower_controller_kwargs}
    return upper_decorator, lower_decorator, controller_kwargs, dataset


def _create_decorator(
    details: Tuple[Type[Decorator], DecoratorDetails],
) -> Tuple[Decorator, Dict[str, Any], Optional[Dataset]]:
    """Unpack decorator details."""
    decorator_class, all_decorator_kwargs = details
    decorator = decorator_class(ignore_all=True, **all_decorator_kwargs.get("decorator", {}))
    return decorator, all_decorator_kwargs.get("controller", {}), all_decorator_kwargs.get("dataset", None)


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
def test_combinations(
    upper_details: Tuple[Type[Decorator], DecoratorDetails], lower_details: Tuple[Type[Decorator], DecoratorDetails]
) -> None:
    """
    For each decorator combination, check that basic usage doesn't throw any unexpected errors.

    In particular, we:

    - Decorate the controller class
    - Instantiate the controller class
    - Fit the controller class
    - For non-classifiers, we then:
      - Take the posterior over some test points
      - Generate a prediction from that posterior
      - Generate a confidence interval from that posterior
      - Take the fuzzy posterior over some test points
      - Generate a prediction from that fuzzy posterior
      - Generate a confidence interval from that fuzzy posterior
    - For classifiers, we instead:
      - Classify some test points
      - Perform fuzzy classification on the test points

    and check that none of the above operations raise any unexpected errors.
    """
    upper_decorator, lower_decorator, controller_kwargs, dataset = _initialise_decorator_pair(
        upper_details, lower_details
    )

    combination = (type(upper_decorator), type(lower_decorator))
    expected_warning_class, expected_warning_message = EXPECTED_COMBINATION_APPLY_WARNINGS.get(
        combination, (None, None)
    )
    expected_error_class, expected_error_message = EXPECTED_COMBINATION_APPLY_ERRORS.get(combination, (None, None))
    with maybe_warns(expected_warning_class, expected_warning_message), maybe_throws(
        expected_error_class, expected_error_message
    ):
        try:
            controller_class = upper_decorator(lower_decorator(GaussianGPController))
        except (MissingRequirementsError, TopmostDecoratorError) as exc:
            if expected_error_class is not None and not isinstance(exc, expected_error_class):
                raise AssertionError(
                    f"Expected {expected_error_class.__name__} to be raised, but got {type(exc).__name__}."
                ) from exc
            return
    if expected_error_class is not None:
        return

    assert dataset is not None
    final_kwargs = {
        "train_x": dataset.train_x,
        "train_y": dataset.train_y,
        "y_std": dataset.train_y_std,
        "kernel_class": ScaledRBFKernel,
        "rng": get_default_rng(),
    }

    combination_controller_kwargs = COMBINATION_CONTROLLER_KWARGS.get(combination, {})
    final_kwargs.update(controller_kwargs)
    final_kwargs.update(combination_controller_kwargs)

    expected_error_class, expected_error_message = EXPECTED_COMBINATION_INIT_ERRORS.get(combination, (None, None))
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
        # check that classification doesn't throw any errors
        controller.classify_points(dataset.test_x)

        # Lower the number of MC samples to speed up testing. We don't care about any kind of accuracy here,
        # so just pick the minimum number that doesn't cause numerical errors.
        with patch.object(MonteCarloPosteriorCollection, "INITIAL_NUMBER_OF_SAMPLES", 25):
            # check that fuzzy classification doesn't throw any errors
            if isinstance(lower_decorator, DirichletKernelMulticlassClassification) or isinstance(
                upper_decorator, DirichletKernelMulticlassClassification
            ):
                # TODO: This test fails as the distribution covariance_matrix is the wrong shape.
                # https://github.com/gchq/Vanguard/issues/288
                pytest.skip("`classify_fuzzy_points` currently fails due to incorrect distribution covariance_matrix")
            controller.classify_fuzzy_points(dataset.test_x, dataset.test_x_std)
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

        # Lower the number of MC samples to speed up testing. Again, we don't care about accuracy here, so just pick
        # the # minimum number that doesn't cause numerical errors.
        with patch.object(MonteCarloPosteriorCollection, "INITIAL_NUMBER_OF_SAMPLES", 4):
            fuzzy_posterior = controller.posterior_over_fuzzy_point(dataset.test_x, dataset.test_x_std)

        try:
            fuzzy_posterior.prediction()
        except TypeError as exc:
            if isinstance(upper_decorator, SetWarp) or isinstance(lower_decorator, SetWarp):
                assert str(exc) == "The mean and covariance of a warped GP cannot be computed exactly."
            else:
                raise

        # Lower the number of MC samples to speed up testing. Again, we don't care about accuracy here, so just pick
        # the minimum number that doesn't cause numerical errors.
        with patch.object(MonteCarloPosteriorCollection, "_decide_mc_num_samples", lambda *_: 4):
            fuzzy_posterior.confidence_interval(dataset.significance)
