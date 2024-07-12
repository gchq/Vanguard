"""
Tests for learning functionality that is not covered elsewhere.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from tests.cases import get_default_rng
from vanguard.datasets.classification import BinaryStripeClassificationDataset
from vanguard.datasets.synthetic import SyntheticDataset
from vanguard.kernels import ScaledRBFKernel
from vanguard.learning import LearnYNoise, _process_y_std
from vanguard.vanilla import GaussianGPController


class TestLearning(unittest.TestCase):
    """
    Tests for usage of the LearnYNoise decorator and associated functionality.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.rng = get_default_rng()
        self.dataset = SyntheticDataset(rng=self.rng)
        self.classification_dataset = BinaryStripeClassificationDataset(
            num_train_points=100, num_test_points=200, rng=self.rng
        )

    def test_no_train_x(self) -> None:
        """Test how LearnYNoise handles the input train_x being missing upon creation."""

        @LearnYNoise()
        class LearnNoiseController(GaussianGPController):
            pass

        # The processing of input arguments done within the decorator catches missing train_x inputs
        # before we hit the line in question for all typical decorators. The extra check later in the
        # initialisation appears to be for specific decorators that could one day be used. As a result
        # we mock the output of the initial check to reach the secondary check.
        with patch("vanguard.decoratorutils.process_args") as mock_process_args:
            mock_process_args.return_value = {
                "self": MagicMock(),
                "rng": self.rng,
                "train_y": self.dataset.train_y,
                "y_std": self.dataset.test_y_std,
            }

            with self.assertRaises(RuntimeError):
                # pylint: disable-next=no-value-for-parameter
                LearnNoiseController(
                    train_y=self.dataset.train_y,
                    kernel_class=ScaledRBFKernel,
                    y_std=self.dataset.train_y_std,
                    rng=self.rng,
                )

    def test_process_y_std_multi_dimensional(self) -> None:
        """Test conversion of y_std with _process_y_std."""
        # Setup inputs
        device = torch.device("cpu")
        y_std = np.array([[0.5, 0.6, 0.7], [5.0, 6.0, 7.0]])

        # We expect a conversion to a torch tensor, with floating point data and sent to the provided device
        expected_result = torch.tensor([[0.5, 0.6, 0.7], [5.0, 6.0, 7.0]], dtype=float, device=device)

        # Call the function and verify output
        result = _process_y_std(y_std=y_std, shape=(2, 3), dtype=float, device=device)
        torch.testing.assert_allclose(result, expected_result)
