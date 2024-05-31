"""
Contains the ClassificationTestCase class.
"""
import unittest
from typing import Union

import numpy as np
import numpy.typing
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean


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

    def __init__(self, batch_shape: torch.Size):
        batch_shape = batch_shape if isinstance(batch_shape, torch.Size) else torch.Size([batch_shape])
        super().__init__(batch_shape=batch_shape)


class ClassificationTestCase(unittest.TestCase):
    """
    A base class for classification tests.
    """

    @staticmethod
    def assertPredictionsEqual(
        x: numpy.typing.NDArray[np.floating], y: numpy.typing.NDArray[np.floating], delta: Union[float, int] = 0
    ) -> None:
        """
        Assert true if predictions are correct.

        :param x: The first set of predictions.
        :param y: The second set of predictions.
        :param delta: The proportion of elements which can be outside of the interval.
        """
        if x.shape != y.shape:
            raise RuntimeError(f"Shape {x.shape} does not match {y.shape}")
        number_incorrect = np.sum(x != y)
        proportion_incorrect = number_incorrect / len(x)
        if proportion_incorrect > delta:
            error_message = (
                f"Incorrect predictions: {number_incorrect} / {len(x)} "
                f"({100 * proportion_incorrect:.2f}%) -- delta = {100 * delta:.2f}%"
            )
            raise AssertionError(error_message) from None
