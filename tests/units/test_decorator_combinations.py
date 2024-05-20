"""
Tests for the pairwise combinations of decorators.
"""

import itertools
import re
import unittest
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import BernoulliLikelihood, DirichletClassificationLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import VariationalELBO

from vanguard.base import GPController
from vanguard.classification import BinaryClassification, DirichletMulticlassClassification
from vanguard.datasets import Dataset
from vanguard.datasets.classification import MulticlassGaussianClassificationDataset
from vanguard.datasets.synthetic import SyntheticDataset, complicated_f, simple_f
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

    pass


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
        "dataset": MulticlassGaussianClassificationDataset(num_train_points=10, num_test_points=10, num_classes=4),
    },
    Distributed: {"decorator": {"n_experts": 3}, "controller": {}},
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
        "dataset": SyntheticDataset([simple_f, complicated_f]),
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
        "decorator": {"n_inducing_points": 20},
        "controller": {
            "likelihood_class": FixedNoiseGaussianLikelihood,
            "marginal_log_likelihood_class": VariationalELBO,
        },
    },
}

COMBINATION_CONTROLLER_KWARGS = {
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

EXPECTED_COMBINATION_FIT_ERRORS = {
    (VariationalInference, Multitask): (RuntimeError, ".* may not be the correct choice for a variational strategy"),
}


class CombinationTests(unittest.TestCase):
    """
    Tests for the pairwise combinations of decorators.
    """

    def test_combinations(self) -> None:
        """Shouldn't throw any errors."""
        for upper_decorator, lower_decorator, controller_kwargs, dataset in self._yield_initialised_decorators():
            with self.subTest(upper=type(upper_decorator).__name__, lower=type(lower_decorator).__name__):
                try:
                    controller_class = upper_decorator(lower_decorator(GaussianGPController))
                except (MissingRequirementsError, TopmostDecoratorError):
                    continue

                final_kwargs = {
                    "train_x": dataset.train_x,
                    "train_y": dataset.train_y,
                    "y_std": dataset.train_y_std,
                    "kernel_class": ScaledRBFKernel,
                }

                combination = (type(upper_decorator), type(lower_decorator))
                combination_controller_kwargs = COMBINATION_CONTROLLER_KWARGS.get(combination, {})
                final_kwargs.update(controller_kwargs)
                final_kwargs.update(combination_controller_kwargs)

                try:
                    controller = controller_class(**final_kwargs)
                except Exception as error:
                    self.fail(f"Could not initialise: {error}")

                try:
                    controller.fit(1)
                except Exception as error:
                    try:
                        expected_error_class, expected_error_message = EXPECTED_COMBINATION_FIT_ERRORS[combination]
                    except KeyError:
                        self.fail(f"Could not train: {error}")

                    if type(error) != expected_error_class:
                        self.fail(f"Expected {expected_error_class} but got: {error}")
                    elif not re.match(expected_error_message, str(error)):
                        self.fail(f"Expected error with the message {expected_error_message!r} but got: {error}")
                    else:
                        return

                try:
                    posterior = controller.posterior_over_point(dataset.test_x)
                except Exception as error:
                    if not hasattr(controller, "classify_points"):
                        self.fail(f"Could not get posterior: {error}")
                else:
                    try:
                        posterior.prediction()
                    except Exception as error:
                        if isinstance(upper_decorator, SetWarp) or isinstance(lower_decorator, SetWarp):
                            pass
                        else:
                            self.fail(f"Could not predict: {error}")

                    try:
                        posterior.confidence_interval(dataset.significance)
                    except Exception as error:
                        self.fail(f"Could not predict: {error}")

                try:
                    fuzzy_posterior = controller.posterior_over_fuzzy_point(dataset.test_x, dataset.test_x_std)
                except Exception as error:
                    if not hasattr(controller, "classify_points"):
                        self.fail(f"Could not get posterior: {error}")
                else:
                    try:
                        fuzzy_posterior.prediction()
                    except Exception as error:
                        if isinstance(upper_decorator, SetWarp) or isinstance(lower_decorator, SetWarp):
                            pass
                        else:
                            self.fail(f"Could not predict: {error}")

                    try:
                        fuzzy_posterior.confidence_interval(dataset.significance)
                    except Exception as error:
                        self.fail(f"Could not predict: {error}")

    def _yield_initialised_decorators(self) -> None:
        """Yield pairs of initialised decorators."""
        for upper_decorator_details, lower_decorator_details in itertools.permutations(DECORATORS.items(), r=2):
            upper_decorator, upper_controller_kwargs, upper_dataset = self._create_decorator(upper_decorator_details)
            lower_decorator, lower_controller_kwargs, lower_dataset = self._create_decorator(lower_decorator_details)

            combination = (type(upper_decorator), type(lower_decorator))
            reversed_combination = (type(lower_decorator), type(upper_decorator))
            if combination in EXCLUDED_COMBINATIONS or reversed_combination in EXCLUDED_COMBINATIONS:
                continue

            if upper_dataset and lower_dataset:
                raise RuntimeError(
                    f"Cannot combine {type(upper_decorator).__name__} and "
                    f"{type(lower_decorator).__name__}: two datasets have been passed."
                )

            dataset = upper_dataset or lower_dataset or SyntheticDataset(n_train_points=20, n_test_points=10)

            controller_kwargs = {**upper_controller_kwargs, **lower_controller_kwargs}
            yield upper_decorator, lower_decorator, controller_kwargs, dataset

    @staticmethod
    def _create_decorator(details: Tuple[Callable, Dict[str, Any]]) -> Tuple[Callable, ControllerT, Optional[Dataset]]:
        """Unpack decorator details."""
        decorator_class, all_decorator_kwargs = details
        decorator = decorator_class(ignore_all=True, **all_decorator_kwargs["decorator"])
        return decorator, all_decorator_kwargs["controller"], all_decorator_kwargs.get("dataset", None)
