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
Tests for the pairwise combinations of decorators.
"""

import itertools
from typing import Any, Optional, TypeVar
from unittest.mock import patch

import numpy as np
import pytest
import sklearn
import torch
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import BernoulliLikelihood, DirichletClassificationLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from typing_extensions import TypedDict

from tests.cases import assert_not_warns, disable_warnings, get_default_rng, maybe_throws, maybe_warns

# not super happy about importing HigherRankKernel/HigherRankMean from another test file - these should probably be
# moved to some more central location
# TODO: replace these with something more general?
# https://github.com/gchq/Vanguard/issues/387
from tests.units.test_features import HigherRankKernel, HigherRankMean
from vanguard.base import GPController
from vanguard.base.posteriors import MonteCarloPosteriorCollection
from vanguard.classification import BinaryClassification, CategoricalClassification, DirichletMulticlassClassification
from vanguard.classification.kernel import DirichletKernelMulticlassClassification
from vanguard.classification.likelihoods import (
    DirichletKernelClassifierLikelihood,
    GenericExactMarginalLogLikelihood,
    MultitaskBernoulliLikelihood,
)
from vanguard.datasets import Dataset
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.datasets.synthetic import HigherRankSyntheticDataset, SyntheticDataset, complicated_f, simple_f
from vanguard.decoratorutils import Decorator, TopMostDecorator
from vanguard.decoratorutils.errors import BadCombinationWarning, OverwrittenMethodWarning, UnexpectedMethodWarning
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
from vanguard.utils import compose
from vanguard.vanilla import GaussianGPController
from vanguard.variational import VariationalInference
from vanguard.warps import SetInputWarp, SetWarp, warpfunctions

T = TypeVar("T")

# Default dataset that is used when neither decorator specifies a test dataset.
DEFAULT_DATASET = SyntheticDataset(n_train_points=20, n_test_points=2, rng=get_default_rng())


@BayesianHyperparameters()
class TestHierarchicalKernel(RBFKernel):
    """A kernel to test the `HierarchicalHyperparameters` decorators with."""


class OneHotMulticlassGaussianClassificationDataset(MulticlassGaussianClassificationDataset):
    """
    Synthetic dataset for use with `CategoricalClassification`.

    This is just a thin wrapper around `MulticlassGaussianClassificationDataset` that performs one-hot encoding on
    its y-values.
    """

    def __init__(self, num_train_points: int, num_test_points: int, num_classes: int, rng: np.random.Generator):
        """Initialise the dataset."""
        super().__init__(
            num_train_points=num_train_points, num_test_points=num_test_points, num_classes=num_classes, rng=rng
        )
        self.train_y = torch.as_tensor(
            sklearn.preprocessing.LabelBinarizer().fit_transform(self.train_y.detach().cpu().numpy())
        )
        self.test_y = torch.as_tensor(
            sklearn.preprocessing.LabelBinarizer().fit_transform(self.test_y.detach().cpu().numpy())
        )


class RequirementDetails(TypedDict, total=False):
    """Type hint for the sub-decorator details specification in `DECORATORS`."""

    decorator: dict[str, Any]
    """Keyword arguments to pass to the decorator."""
    controller: dict[str, Any]
    """Additional keyword arguments to pass to the `GPController` on instantiation."""


class DecoratorDetails(TypedDict, total=False):
    """Type hint for the decorator details specifications in `DECORATORS`."""

    decorator: dict[str, Any]
    """Keyword arguments to pass to the decorator."""
    controller: dict[str, Any]
    """Additional keyword arguments to pass to the `GPController` on instantiation."""
    dataset: Dataset
    """
    Specifies a dataset for the `GPController` to be fitted and tested on.

    If unspecified, the dataset for the other decorator is used, or `DEFAULT_DATASET`, if the other decorator doesn't
    specify a dataset.

    Can be overridden for specific decorator combinations in `DATASET_CONFLICT_OVERRIDES`.
    """
    # TODO(rg): Remove the requirements specification from here if it gets added to the Decorator framework
    # https://github.com/gchq/Vanguard/issues/381
    requirements: dict[type[Decorator], RequirementDetails]
    """
    Specifies additional decorators that this decorator requires.

    These are applied to the `GPController` before this decorator.
    """


class EmptyDecorator(Decorator):
    """Placeholder decorator to allow testing each other decorator on its own."""

    def __init__(self, **kwargs) -> None:
        """
        Initialise the placeholder decorator.

        `ignore_all=True` is always passed, since this decorator is meant to be a no-op.
        """
        kwargs["ignore_all"] = True
        super().__init__(framework_class=GPController, required_decorators={}, **kwargs)

    def _decorate_class(self, cls: T) -> T:
        """Return the class unaltered."""
        return cls


# Details for each of the decorators under test. See `DecoratorDetails` for documentation on each key.
DECORATORS: dict[type[Decorator], DecoratorDetails] = {
    EmptyDecorator: {},
    BinaryClassification: {
        "controller": {
            "y_std": 0.0,
            "likelihood_class": BernoulliLikelihood,
            "marginal_log_likelihood_class": VariationalELBO,
        },
        "requirements": {VariationalInference: {"decorator": {"n_inducing_points": 5}}},
    },
    DirichletMulticlassClassification: {
        "decorator": {"num_classes": 4},
        "controller": {
            "likelihood_class": DirichletClassificationLikelihood,
            "likelihood_kwargs": {"learn_additional_noise": True},
            "kernel_class": ScaledRBFKernel,
            "kernel_kwargs": {"batch_shape": (4,)},
            "y_std": 0.0,
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
        "controller": {"kernel_class": HigherRankKernel, "mean_class": HigherRankMean},
        "dataset": HigherRankSyntheticDataset(n_train_points=10, n_test_points=4, rng=get_default_rng()),
    },
    DisableStandardScaling: {},
    CategoricalClassification: {
        "decorator": {"num_classes": 4},
        "controller": {
            "likelihood_class": MultitaskBernoulliLikelihood,
            "marginal_log_likelihood_class": VariationalELBO,
            "y_std": 0.0,
        },
        "dataset": OneHotMulticlassGaussianClassificationDataset(
            num_train_points=20, num_test_points=4, num_classes=4, rng=get_default_rng()
        ),
        "requirements": {
            Multitask: {"decorator": {"num_tasks": 4}},
            VariationalInference: {"decorator": {}},
        },
    },
    Distributed: {"decorator": {"n_experts": 3, "rng": get_default_rng()}},
    VariationalHierarchicalHyperparameters: {
        "decorator": {"num_mc_samples": 13},
        "controller": {"kernel_class": TestHierarchicalKernel},
    },
    LaplaceHierarchicalHyperparameters: {
        "decorator": {"num_mc_samples": 13},
        "controller": {"kernel_class": TestHierarchicalKernel},
    },
    LearnYNoise: {},
    NormaliseY: {},
    Multitask: {
        "decorator": {"num_tasks": 2},
        "controller": {"likelihood_class": FixedNoiseMultitaskGaussianLikelihood},
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

# (upper, lower) -> kwargs
# Additional keyword arguments to provide to the `GPController` on instantiation for specific decorator combinations
COMBINATION_CONTROLLER_KWARGS: dict[tuple[type[Decorator], type[Decorator]], dict[str, Any]] = {
    (VariationalInference, DirichletMulticlassClassification): {"likelihood_class": DirichletClassificationLikelihood},
    (VariationalInference, CategoricalClassification): {"likelihood_class": MultitaskBernoulliLikelihood},
    (VariationalInference, BinaryClassification): {"likelihood_class": BernoulliLikelihood},
    (Multitask, VariationalInference): {"likelihood_class": FixedNoiseMultitaskGaussianLikelihood},
}

# Combinations that are not tested at all. This is matched against independent of order, so if (A, B) is listed,
# neither (A, B) nor (B, A) is tested. Note that these are not skipped, they are just not provided as parameter
# combinations in the first place.
# TODO(rg): Ideally, this set should be empty; everything in here should be a temporary measure. If a pair of decorators
#  are incompatible, we should raise an exception saying so, and then we should test for that exception here.
# https://github.com/gchq/Vanguard/issues/386
EXCLUDED_COMBINATIONS = {
    # Multitask classification generally doesn't work:
    # TODO(rg): these mainly fail due to dataset conflicts - we should provide datasets that work with these
    #  combinations
    # https://github.com/gchq/Vanguard/issues/385
    (Multitask, BinaryClassification),  # Likelihood contradiction
    (Multitask, CategoricalClassification),  # Multiple datasets - unnecessary as Multitask is already a requirement
    (Multitask, DirichletMulticlassClassification),  # Multiple datasets
    (Multitask, DirichletKernelMulticlassClassification),  # Multiple datasets
    # Nor does classification with variational inference:
    # TODO(rg): if this is an accepted incompatibility, we should raise an exception saying so
    # https://github.com/gchq/Vanguard/issues/386
    (VariationalInference, DirichletKernelMulticlassClassification),  # MLL/likelihood class contradiction
    # Conflicts with Distributed:
    # TODO(rg): if this is an accepted incompatibility, we should raise an exception saying so
    # https://github.com/gchq/Vanguard/issues/386
    (Distributed, Multitask),  # Cannot aggregate multitask predictions (shape errors)
    # Can't aggregate multitask predictions:
    # TODO(rg): Commenting out either the (VHH, DMC) or (LHH, DMC) pair below causes several unrelated combinations
    #  DirichletMulticlassClassification to fail. This indicates a failure of test isolation.
    # https://github.com/gchq/Vanguard/issues/378
    (VariationalHierarchicalHyperparameters, DirichletMulticlassClassification),
    (VariationalHierarchicalHyperparameters, DirichletKernelMulticlassClassification),
    (LaplaceHierarchicalHyperparameters, DirichletMulticlassClassification),
    (LaplaceHierarchicalHyperparameters, DirichletKernelMulticlassClassification),
    # TODO(rg): these fail with AttributeError: 'Bernoulli' object has no attribute 'covariance_matrix'
    # https://github.com/gchq/Vanguard/issues/382
    (BinaryClassification, VariationalHierarchicalHyperparameters),
    (BinaryClassification, LaplaceHierarchicalHyperparameters),
    (CategoricalClassification, VariationalHierarchicalHyperparameters),
    (CategoricalClassification, LaplaceHierarchicalHyperparameters),
    # TODO(rg): We should provide some datasets that work with these combinations.
    # https://github.com/gchq/Vanguard/issues/385
    # HigherRankFeatures has dataset conflicts with several other decorators:
    (HigherRankFeatures, DirichletMulticlassClassification),  # Two datasets
    (HigherRankFeatures, DirichletKernelMulticlassClassification),  # Two datasets
    (HigherRankFeatures, CategoricalClassification),  # Two datasets
    (HigherRankFeatures, Multitask),  # Two datasets
    # TEMPORARY - TO FIX:
    # TODO(rg): Fails with an "index out of bounds" error - seems to be because the warp function moves the class
    #  indices out of the expected range. `DirichletMulticlassClassification` seems to work fine though. Unsure on
    #  `BinaryClassification` or `CategoricalClassification` - these two aren't tested with `SetWarp` due to a
    #  `MissingRequirementsError`.
    # https://github.com/gchq/Vanguard/issues/376
    (DirichletKernelMulticlassClassification, SetWarp),
    # TODO(rg): Fails due to shape mismatch whichever one is on top. When VHH/LHH is on top of HRF this makes sense,
    #  but the other way around should probably work. Will require a custom @BayesianHyperparameters higher-rank
    #  kernel class.
    # https://github.com/gchq/Vanguard/issues/375
    (HigherRankFeatures, VariationalHierarchicalHyperparameters),
    (HigherRankFeatures, LaplaceHierarchicalHyperparameters),
}

# Combinations that are not tested at all in batch mode. Note that these are not skipped, they are just not provided
# as parameter combinations in the first place. This list is checked after `EXCLUDED_COMBINATIONS`, so anything that
# appears in that list is also excluded. Unlike `EXCLUDED_COMBINATIONS`, this is a set of ordered (upper, lower) pairs.
BATCH_EXCLUDED_COMBINATIONS = {
    # TODO: These are all excluded as a *temporary measure*. Eventually this list should be empty. For each pair,
    #  either raise an informative exception we can test for, or ensure that it runs without error.
    # https://github.com/gchq/Vanguard/issues/386
    (Distributed, DirichletMulticlassClassification),
    (Distributed, SetWarp),
    (VariationalHierarchicalHyperparameters, Multitask),
    (LaplaceHierarchicalHyperparameters, Multitask),
    (Multitask, VariationalHierarchicalHyperparameters),
    (Multitask, LaplaceHierarchicalHyperparameters),
    (Multitask, Multitask),
}

# (upper, lower) -> (error type, message regex)
# Errors we expect to be raised on initialisation of the decorated class.
EXPECTED_COMBINATION_INIT_ERRORS: dict[tuple[type[Decorator], type[Decorator]], tuple[type[Exception], str]] = {
    (NormaliseY, DirichletMulticlassClassification): (
        TypeError,
        "NormaliseY should not be used above classification decorators.",
    ),
}

# (upper, lower) -> (error type, message regex)
# Errors we expect to be raised on decorator application.
EXPECTED_COMBINATION_APPLY_ERRORS: dict[tuple[type[Decorator], type[Decorator]], tuple[type[Exception], str]] = {
    (Distributed, HigherRankFeatures): (
        TypeError,
        ".* cannot handle higher-rank features. Consider moving the `@Distributed` decorator "
        "below the `@HigherRankFeatures` decorator.",
    ),
    # Can only use one hyperparameter decorator at once:
    **{
        (upper, lower): (
            TypeError,
            f"This class is already decorated with `{lower.__name__}`. "
            f"Please use only one hierarchical hyperparameters decorator at once.",
        )
        for upper, lower in itertools.product(
            [VariationalHierarchicalHyperparameters, LaplaceHierarchicalHyperparameters], repeat=2
        )
    },
    # Can only use one classification decorator at a time:
    **{
        (upper, lower): (
            TypeError,
            "This class is already decorated with a classification decorator. "
            "Please use only one classification decorator at once.",
        )
        for upper, lower in itertools.product(
            [
                BinaryClassification,
                CategoricalClassification,
                DirichletMulticlassClassification,
                DirichletKernelMulticlassClassification,
            ],
            repeat=2,
        )
    },
}

# (upper, lower) -> (warning type, message regex)
# Warnings we expect to be raised on decorator application.
EXPECTED_COMBINATION_APPLY_WARNINGS: dict[tuple[type[Decorator], type[Decorator]], tuple[type[Warning], str]] = {
    **{
        (NormaliseY, lower): (
            BadCombinationWarning,
            "NormaliseY should not be used above classification decorators - this may lead to unexpected behaviour.",
        )
        for lower in [
            BinaryClassification,
            CategoricalClassification,
            DirichletMulticlassClassification,
            DirichletKernelMulticlassClassification,
        ]
    },
    **{
        (VariationalInference, lower): (
            BadCombinationWarning,
            "Multiple instances of `@VariationalInference` not supported."
            " Please only apply one instance of `@VariationalInference` at once.",
        )
        for lower in [VariationalInference, BinaryClassification, CategoricalClassification]
    },
    (HigherRankFeatures, HigherRankFeatures): (
        BadCombinationWarning,
        "Multiple instances of `@HigherRankFeatures` not supported."
        " Please only apply one instance of `@HigherRankFeatures` at once.",
    ),
    (Multitask, Multitask): (
        BadCombinationWarning,
        "Multiple instances of `@Multitask` not supported. Please only apply one instance of `@Multitask` at once.",
    ),
}

# (upper, lower) -> (error type, message regex)
# Errors we expect to be raised when calling controller.fit().
EXPECTED_COMBINATION_FIT_ERRORS: dict[tuple[type[Decorator], type[Decorator]], tuple[type[Exception], str]] = {
    (VariationalInference, Multitask): (RuntimeError, ".* may not be the correct choice for a variational strategy"),
}

# (upper, lower) -> dataset
# Combinations for which we ignore the normal error raised when we try and pass two datasets, and instead provide a
# replacement dataset.
DATASET_CONFLICT_OVERRIDES = {
    # Ignore combinations of classification decorators which might have conflicting dataset requirements - applying
    # two classification decorators raises an error which we want to check for. It doesn't matter that we provide
    # `None` here - the dataset is only used for initialisation, and these should fail on decorator application.
    **{
        (upper, lower): None
        for upper, lower in itertools.product(
            [
                CategoricalClassification,
                DirichletMulticlassClassification,
                DirichletKernelMulticlassClassification,
            ],
            repeat=2,
        )
    }
}


def _initialise_decorator_pair(
    upper_decorator_details: tuple[type[Decorator], DecoratorDetails],
    lower_decorator_details: tuple[type[Decorator], DecoratorDetails],
    *,
    batch_mode: bool,
) -> tuple[Decorator, list[Decorator], Decorator, list[Decorator], list[Decorator], dict[str, Any], Optional[Dataset]]:
    """
    Initialise a pair of decorators for testing.

    :param upper_decorator_details: (key, value) entry from `DECORATORS` for the upper decorator.
    :param upper_decorator_details: (key, value) entry from `DECORATORS` for the lower decorator.
    :param batch_mode: True if the controller will be created in batch mode. Ensures `VariationalInference` decorator
        is present.
    :return: Tuple (upper_decorator, upper_requirement_decorators, lower_decorator, lower_requirement_decorators,
        batch_decorators, controller_kwargs, dataset):
        - `upper_decorator`: the instantiated upper decorator
        - `upper_requirement_decorators`: list of instantiated decorators to be applied before `upper_decorator` to
            fulfil its requirements
        - `lower_decorator`: the instantiated upper decorator
        - `lower_requirement_decorators`: list of instantiated decorators to be applied before `lower_decorator` to
            fulfil its requirements
        - `batch_decorators`: possibly contains an instantiated `VariationalInference` decorator if running in batch
            mode and one is not already present in the previous tuple elements
        - `controller_kwargs`: additional keyword arguments to provide to the `GPController` on instantiation. In case
            multiple decorators provide values for the same keyword argument, if a value is given in
            `COMBINATION_CONTROLLER_KWARGS`, that value is used; if not the value from the topmost decorator is used.
        - `dataset`: dataset to provide to the `GPController` on instantiation and test with
    """
    upper_decorator, upper_requirement_decorators, upper_controller_kwargs, upper_dataset = _create_decorator(
        upper_decorator_details
    )
    lower_decorator, lower_requirement_decorators, lower_controller_kwargs, lower_dataset = _create_decorator(
        lower_decorator_details
    )

    if (type(upper_decorator), type(lower_decorator)) in DATASET_CONFLICT_OVERRIDES:
        # Decorator application *must* fail, so passing dataset=None doesn't matter as we'll never reach initialisation
        dataset = DATASET_CONFLICT_OVERRIDES[type(upper_decorator), type(lower_decorator)]
    elif upper_decorator_details == lower_decorator_details:
        # The same details means the dataset is identical, so pass it if present, or a default if not
        dataset = upper_dataset or DEFAULT_DATASET
    elif upper_dataset and lower_dataset:
        # Passing two datasets is ambiguous!
        raise RuntimeError(
            f"Cannot combine {type(upper_decorator).__name__} and "
            f"{type(lower_decorator).__name__}: two datasets have been passed."
        )
    else:
        # Pass whichever dataset we have, or a default.
        dataset = upper_dataset or lower_dataset or DEFAULT_DATASET

    # Avoid duplicate decorators from requirements. Lower requirements take precedence over upper requirements if there
    # are duplicates, to ensure that both the upper and lower decorator can see them.
    upper_requirement_decorators = [
        decorator
        for decorator in upper_requirement_decorators
        if not isinstance(decorator, type(lower_decorator))
        and not any(
            isinstance(decorator, type(lower_requirement_decorator))
            for lower_requirement_decorator in lower_requirement_decorators
        )
    ]

    if batch_mode and not any(
        isinstance(decorator, VariationalInference)
        for decorator in [
            upper_decorator,
            *upper_requirement_decorators,
            lower_decorator,
            *lower_requirement_decorators,
        ]
    ):
        # then we need to add variational inference ourselves
        variational_decorator, _, controller_kwargs, _ = _create_decorator(
            (VariationalInference, DECORATORS[VariationalInference])
        )
        batch_decorators = [variational_decorator]
    else:
        batch_decorators = []
        controller_kwargs = {}

    # For controller arguments, ones on higher decorators override those on lower decorators
    controller_kwargs.update(lower_controller_kwargs)
    controller_kwargs.update(upper_controller_kwargs)
    return (
        upper_decorator,
        upper_requirement_decorators,
        lower_decorator,
        lower_requirement_decorators,
        batch_decorators,
        controller_kwargs,
        dataset,
    )


def _create_decorator(
    details: tuple[type[Decorator], DecoratorDetails],
) -> tuple[Decorator, list[Decorator], dict[str, Any], Optional[Dataset]]:
    """
    Unpack decorator details.

    :param details: (key, value) entry from `DECORATORS` for the decorator.
    :return: Tuple (decorator, requirement_decorators, controller_kwargs, optional dataset):
        - `decorator`: the instantiated decorator
        - `requirement_decorators`: list of instantiated decorators to be applied before `decorator` to fulfil
            requirements
        - `controller_kwargs`: additional keyword arguments to provide to the `GPController` on instantiation
        - `dataset`: dataset to test with, or :data:`None` to defer to the other decorator or to a default
    """
    decorator_class, decorator_details = details
    decorator = decorator_class(**decorator_details.get("decorator", {}))
    controller_kwargs = {}

    requirement_decorators = []
    requirements_details = decorator_details.get("requirements", {})
    for required_decorator_class, required_decorator_details in requirements_details.items():
        # For controller arguments, ones on higher decorators override those on lower decorators
        requirement_decorators.append(required_decorator_class(**required_decorator_details.get("decorator", {})))
        controller_kwargs.update(required_decorator_details.get("controller", {}))

    # ...and the ones from the main decorator are given priority over those from requirement decorators
    controller_kwargs.update(decorator_details.get("controller", {}))

    return decorator, requirement_decorators, controller_kwargs, decorator_details.get("dataset", None)


@pytest.mark.parametrize(
    "upper_details, lower_details",
    [
        pytest.param(
            upper_details,
            lower_details,
            id=(
                f"Upper: {upper_details[0].__name__}-Lower: {lower_details[0].__name__}"
                if lower_details[0] is not EmptyDecorator
                else f"Only {upper_details[0].__name__}"
            ),
        )
        for upper_details, lower_details in itertools.product(DECORATORS.items(), repeat=2)
        # Don't test combinations which we've excluded above
        if (upper_details[0], lower_details[0]) not in EXCLUDED_COMBINATIONS
        and (lower_details[0], upper_details[0]) not in EXCLUDED_COMBINATIONS
        # NoDecorator should only be on bottom, to avoid cluttering the test log
        and upper_details[0] is not EmptyDecorator
        # TopMostDecorators must be on top, as the name suggests
        and not issubclass(lower_details[0], TopMostDecorator)
    ],
)
@pytest.mark.parametrize("batch_size", [pytest.param(None, id="full"), pytest.param(2, id="batch")])
def test_combinations(
    upper_details: tuple[type[Decorator], DecoratorDetails],
    lower_details: tuple[type[Decorator], DecoratorDetails],
    batch_size: Optional[int],
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
    (
        upper_decorator,
        upper_requirements,
        lower_decorator,
        lower_requirements,
        batch_requirements,
        controller_kwargs,
        dataset,
    ) = _initialise_decorator_pair(upper_details, lower_details, batch_mode=batch_size is not None)
    all_decorators = [upper_decorator, *upper_requirements, lower_decorator, *lower_requirements, *batch_requirements]

    if batch_size is not None and any(isinstance(d, DirichletKernelMulticlassClassification) for d in all_decorators):
        pytest.skip("DirichletKernelMulticlassClassification is not compatible with VariationalInference")
    if batch_size is not None and (upper_details[0], lower_details[0]) in BATCH_EXCLUDED_COMBINATIONS:
        pytest.skip("Combination is excluded from batch mode testing")

    combination = (type(upper_decorator), type(lower_decorator))
    expected_warning_class, expected_warning_message = EXPECTED_COMBINATION_APPLY_WARNINGS.get(
        combination, (None, None)
    )
    expected_error_class, expected_error_message = EXPECTED_COMBINATION_APPLY_ERRORS.get(combination, (None, None))
    if expected_error_class is not None or expected_warning_class is not None:
        # If we expect some other error or warning, we might also get these warnings too, so ignore them.
        warnings_context = disable_warnings(OverwrittenMethodWarning, UnexpectedMethodWarning)
    else:
        # Otherwise, we shouldn't get any of these spurious warnings.
        warnings_context = assert_not_warns(OverwrittenMethodWarning, UnexpectedMethodWarning)
    with (
        warnings_context,
        maybe_warns(expected_warning_class, expected_warning_message),
        maybe_throws(expected_error_class, expected_error_message),
    ):
        controller_class = compose(all_decorators)(GaussianGPController)
    if expected_error_class is not None:
        return

    if isinstance(upper_decorator, HigherRankFeatures) and isinstance(lower_decorator, HigherRankFeatures):
        # TODO(rg): figure out what to do with this? Do we make the HRF decorator raise an error if applied to a class
        #  already decorated with HRF? Or do we try and provide some appropriate arguments to make this work?
        # https://github.com/gchq/Vanguard/issues/383
        pytest.skip("Needs more work!")

    assert dataset is not None
    final_kwargs = {
        "train_x": dataset.train_x,
        "train_y": dataset.train_y,
        "y_std": dataset.train_y_std,
        "kernel_class": ScaledRBFKernel,
        "rng": get_default_rng(),
        "batch_size": batch_size,
    }

    combination_controller_kwargs = COMBINATION_CONTROLLER_KWARGS.get(combination, {})
    final_kwargs.update(controller_kwargs)
    final_kwargs.update(combination_controller_kwargs)

    # Instantiate the controller
    expected_error_class, expected_error_message = EXPECTED_COMBINATION_INIT_ERRORS.get(combination, (None, None))
    with maybe_throws(expected_error_class, expected_error_message):
        controller = controller_class(**final_kwargs)
    if expected_error_class is not None:
        return

    # Fit the controller
    expected_error_class, expected_error_message = EXPECTED_COMBINATION_FIT_ERRORS.get(combination, (None, None))
    with maybe_throws(expected_error_class, expected_error_message):
        controller.fit(2)
    if expected_error_class is not None:
        return

    # If it's a classifier, check that the classification methods don't throw any unexpected errors
    if hasattr(controller, "classify_points"):
        try:
            controller.classify_points(dataset.test_x)
        except TypeError as exc:
            if any(isinstance(decorator, SetWarp) for decorator in all_decorators):
                assert str(exc) == "The mean and covariance of a warped GP cannot be computed exactly."
            else:
                raise

        # Lower the number of MC samples to speed up testing. We don't care about any kind of accuracy here,
        # so just pick the minimum number that doesn't cause numerical errors.
        with patch.object(
            MonteCarloPosteriorCollection,
            "INITIAL_NUMBER_OF_SAMPLES",
            # TODO(rg): Investigate why CategoricalClassification needs so many more samples?
            # https://github.com/gchq/Vanguard/issues/380
            90 if any(isinstance(decorator, CategoricalClassification) for decorator in all_decorators) else 20,
        ):
            # check that fuzzy classification doesn't throw any errors
            if any(isinstance(decorator, DirichletKernelMulticlassClassification) for decorator in all_decorators):
                # TODO(rg): This test fails as the distribution covariance_matrix is the wrong shape.
                # https://github.com/gchq/Vanguard/issues/288
                pytest.skip("`classify_fuzzy_points` currently fails due to incorrect distribution covariance_matrix")
            try:
                controller.classify_fuzzy_points(dataset.test_x, dataset.test_x_std)
            except TypeError as exc:
                if any(isinstance(decorator, SetWarp) for decorator in all_decorators):
                    assert str(exc) == "The mean and covariance of a warped GP cannot be computed exactly."
                else:
                    raise

    # If not, check that the prediction methods don't throw any unexpected errors
    else:
        try:
            posterior = controller.posterior_over_point(dataset.test_x)
        except RuntimeError as exc:
            if isinstance(upper_decorator, Distributed) and isinstance(
                lower_decorator, VariationalHierarchicalHyperparameters
            ):
                assert str(exc) == "Cannot distribute using this kernel - try using a non-BCM aggregator instead."
                return
            else:
                raise

        try:
            posterior.prediction()
        except TypeError as exc:
            if any(isinstance(decorator, SetWarp) for decorator in all_decorators):
                assert str(exc) == "The mean and covariance of a warped GP cannot be computed exactly."
            else:
                raise

        posterior.confidence_interval(dataset.significance)

        # Lower the number of MC samples to speed up testing. Again, we don't care about accuracy here, so just pick
        # the minimum number that doesn't cause numerical errors.
        with patch.object(MonteCarloPosteriorCollection, "INITIAL_NUMBER_OF_SAMPLES", 4):
            fuzzy_posterior = controller.posterior_over_fuzzy_point(dataset.test_x, dataset.test_x_std)

        try:
            fuzzy_posterior.prediction()
        except TypeError as exc:
            if any(isinstance(decorator, SetWarp) for decorator in all_decorators):
                assert str(exc) == "The mean and covariance of a warped GP cannot be computed exactly."
            else:
                raise

        # Lower the number of MC samples to speed up testing. Again, we don't care about accuracy here, so just pick
        # the minimum number that doesn't cause numerical errors.
        with patch.object(MonteCarloPosteriorCollection, "_decide_mc_num_samples", lambda *_: 4):
            fuzzy_posterior.confidence_interval(dataset.significance)
