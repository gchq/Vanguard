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
Tests for learning functionality that is not covered elsewhere.
"""

import sys
import unittest
from typing import Any, Optional
from unittest.mock import patch

import numpy as np
import pytest
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ZeroMean

from tests.cases import get_default_rng
from vanguard.classification import DirichletMulticlassClassification
from vanguard.datasets.classification import BinaryStripeClassificationDataset, MulticlassGaussianClassificationDataset
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.learning import LearnYNoise, _process_y_std
from vanguard.vanilla import GaussianGPController


class BatchScaledRBFKernel(ScaleKernel):
    """
    The recommended starting place for a kernel.
    """

    def __init__(self, batch_shape: torch.Size) -> None:
        batch_shape = batch_shape if isinstance(batch_shape, torch.Size) else torch.Size([batch_shape])
        super().__init__(RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape)


class BatchScaledMean(ZeroMean):
    """
    A basic mean with batch shape to match the above kernel.
    """

    def __init__(self, batch_shape: torch.Size) -> None:
        batch_shape = batch_shape if isinstance(batch_shape, torch.Size) else torch.Size([batch_shape])
        super().__init__(batch_shape=batch_shape)


class AlteredDirichletClassificationLikelihoodExpectedError(DirichletClassificationLikelihood):
    """
    Define a likelihood class that rejects the argument `learn_additional_noise` for testing.
    """

    def __init__(
        self,
        targets: torch.Tensor,
        alpha_epsilon: float = 0.01,
        learn_additional_noise: Optional[bool] = False,
        batch_shape: torch.Size = torch.Size(),
        dtype: torch.dtype = torch.float,
        **kwargs: Any,
    ):
        """
        Initialise self, and raise a sensible `TypeError` if `learn_additional_noise` is passed as True.
        """
        if learn_additional_noise:
            raise TypeError("__init__() got an unexpected keyword argument 'learn_additional_noise'")
        super().__init__(
            targets,
            alpha_epsilon,
            learn_additional_noise,
            batch_shape,
            dtype,
            **kwargs,
        )


class AlteredDirichletClassificationLikelihoodUnexpectedError(DirichletClassificationLikelihood):
    """
    Define a likelihood class that rejects the argument `learn_additional_noise` for testing.

    The rejection in this case is not an expected refusal of learning noise, and hence should be
    treated differently in the Vanguard internal code.
    """

    def __init__(
        self,
        targets: torch.Tensor,
        alpha_epsilon: float = 0.01,
        learn_additional_noise: Optional[bool] = False,
        batch_shape: torch.Size = torch.Size(),
        dtype: torch.dtype = torch.float,
        **kwargs: Any,
    ):
        """
        Initialise self, and raise a `ValueError` if `learn_additional_noise` is passed as True.
        """
        if learn_additional_noise:
            raise ValueError("Argument 'learn_additional_noise' cannot be set to True.")
        super().__init__(
            targets,
            alpha_epsilon,
            learn_additional_noise,
            batch_shape,
            dtype,
            **kwargs,
        )


class TestLearning(unittest.TestCase):
    """
    Tests for usage of the `LearnYNoise` decorator and associated functionality.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.rng = get_default_rng()
        self.dataset = SyntheticDataset(rng=self.rng)
        self.classification_dataset = BinaryStripeClassificationDataset(
            num_train_points=100, num_test_points=200, rng=self.rng
        )

    def test_no_train_x(self) -> None:
        """Test how `LearnYNoise` handles the input train_x being missing upon creation."""

        @LearnYNoise()
        class LearnNoiseController(GaussianGPController):
            pass

        # The processing of input arguments done within the decorator catches missing train_x inputs
        # before we hit the line in question for all typical decorators. The extra check later in the
        # initialisation appears to be for specific decorators that could one day be used. As a result
        # we mock the output of the initial check to reach the secondary check.
        with patch("vanguard.decoratorutils.process_args") as mock_process_args:
            mock_process_args.return_value = {
                "rng": self.rng,
                "train_y": self.dataset.train_y,
                "y_std": self.dataset.test_y_std,
            }

            with self.assertRaises(RuntimeError):
                # pylint: disable=no-value-for-parameter
                LearnNoiseController(
                    train_y=self.dataset.train_y,
                    kernel_class=ScaledRBFKernel,
                    y_std=self.dataset.train_y_std,
                    rng=self.rng,
                )
                # pylint: enable=no-value-for-parameter

    def test_process_y_std_multi_dimensional(self) -> None:
        """Test conversion of `y_std` with `_process_y_std`."""
        # Setup inputs
        device = torch.device("cpu")
        y_std = np.array([[0.5, 0.6, 0.7], [5.0, 6.0, 7.0]])

        # We expect a conversion to a torch tensor, with floating point data and sent to the provided device
        expected_result = torch.tensor([[0.5, 0.6, 0.7], [5.0, 6.0, 7.0]], dtype=float, device=device)

        # Call the function and verify output
        result = _process_y_std(y_std=y_std, shape=(2, 3), dtype=float, device=device)
        torch.testing.assert_allclose(result, expected_result)

    @pytest.mark.skipif(sys.version_info < (3, 12), reason="requires python3.12 or higher")
    def test_with_noise_learning(self) -> None:
        """
        Test controller creation outcomes with different likelihoods.

        We consider a likelihood that allows training of output noise, one that does not but raises an expected
        error we can handle, and one that raises an unexpected error that we do not directly handle.
        """

        # Define a decorator that one might wish to use `LearnYNoise` with in practice
        @LearnYNoise()
        @DirichletMulticlassClassification(num_classes=4, ignore_methods=("__init__",))
        class DirichletMulticlassClassifier(GaussianGPController):
            """A simple Dirichlet multiclass classifier."""

        # Define a dataset that the above decorator may be used on
        dataset = MulticlassGaussianClassificationDataset(
            num_train_points=150, num_test_points=100, num_classes=4, rng=self.rng
        )

        # Create a controller using `DirichletClassificationLikelihood`, which supports learning the noise in
        # the dataset (that is accepts the parameter learn_additional_noise as a likelihood keyword argument).
        # If this decorator can be successfully created, we are happy with the initial usage.
        DirichletMulticlassClassifier(
            dataset.train_x,
            dataset.train_y,
            y_std=0,
            mean_class=BatchScaledMean,
            kernel_class=BatchScaledRBFKernel,
            likelihood_class=DirichletClassificationLikelihood,
            likelihood_kwargs={"alpha_epsilon": 0.3, "learn_additional_noise": True},
            optim_kwargs={"lr": 0.05},
            kernel_kwargs={"batch_shape": 4},
            mean_kwargs={"batch_shape": 4},
            rng=self.rng,
        )

        # Now, if we use a likelihood class but don't let it accept the keyword argument learn_additional_noise, then
        # it should reach a `TypeError` and avoid trying to learn the noise - but not directly fail creation, just
        # raise a warning
        with self.assertWarnsRegex(
            Warning,
            "Cannot learn additional noise for 'AlteredDirichletClassificationLikelihoodExpectedError'. "
            "Consider removing the 'LearnYNoise' decorator.",
        ):
            DirichletMulticlassClassifier(
                dataset.train_x,
                dataset.train_y,
                y_std=0,
                mean_class=BatchScaledMean,
                kernel_class=BatchScaledRBFKernel,
                likelihood_class=AlteredDirichletClassificationLikelihoodExpectedError,
                likelihood_kwargs={"alpha_epsilon": 0.3, "learn_additional_noise": True},
                optim_kwargs={"lr": 0.05},
                kernel_kwargs={"batch_shape": 4},
                mean_kwargs={"batch_shape": 4},
                rng=self.rng,
            )

        # Finally, if we create a controller with a likelihood class that does not give a `TypeError` when passed
        # `learn_additional_noise`, the code should outright fail, not just warn the user.
        with self.assertRaises(ValueError):
            DirichletMulticlassClassifier(
                dataset.train_x,
                dataset.train_y,
                y_std=0,
                mean_class=BatchScaledMean,
                kernel_class=BatchScaledRBFKernel,
                likelihood_class=AlteredDirichletClassificationLikelihoodUnexpectedError,
                likelihood_kwargs={"alpha_epsilon": 0.3, "learn_additional_noise": True},
                optim_kwargs={"lr": 0.05},
                kernel_kwargs={"batch_shape": 4},
                mean_kwargs={"batch_shape": 4},
                rng=self.rng,
            )
