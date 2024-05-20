"""
Contains the LearningRateFinder class to aid with choosing the largest possible learning rate.
"""
import typing

import matplotlib.pyplot as plt
import numpy as np
from gpytorch.utils.errors import NanError

if typing.TYPE_CHECKING:
    from vanguard.base import GPController


class LearningRateFinder:
    """
    Estimates the best learning rate for a controller/data combination.

    Try an increasing geometric sequence of learning rates for a small number of iterations to find
    the best learning rate (i.e. the largest learning rate giving stable training).
    """

    def __init__(self, controller: "GPController"):
        """
        Initialise self.

        :param controller: An instantiated vanguard GP controller whose learning rate shall be optimised.
        """
        self._controller = controller
        self._learning_rates = []
        self._losses = []

    @property
    def best_learning_rate(self) -> float:
        return self._learning_rates[np.argmin(self._losses)]

    def find(
        self, start_lr: float = 1e-5, end_lr: float = 10, num_divisions: int = 100, max_iterations: int = 20
    ) -> None:
        """
        Try the range of learning rates and record the loss obtained.

        :param start_lr: The smallest learning rate to try.
        :param end_lr: The largest learning rate to try.
        :param num_divisions: The number of learning rates to try.
        :param max_iterations: The top number of iterations of gradient descent to run for each learning rate.
        """
        ratio = np.power(end_lr / start_lr, 1.0 / num_divisions)
        self._learning_rates = [start_lr * ratio**index for index in range(num_divisions)]
        self._losses = [self._run_learning_rate(lr, max_iterations) for lr in self._learning_rates]

    def plot(self, **kwargs) -> None:
        """
        Plot the obtained loss-vs-lr curve.
        """
        plt.plot(self._learning_rates, self._losses, **kwargs)
        plt.xlabel("learning rate")
        plt.ylabel("best loss")
        plt.show()

    def _run_learning_rate(self, lr: float, max_iterations: int) -> float:
        """
        Do the training for a single learning rate.

        If training fails due to NaNs before the max iterations is reached,
        the loss is taken to be infinity.

        :param lr: The learning rate.
        :param max_iterations: The maximum number of gradient descent iterations to run.
        :returns: The loss at the end of training with the given learning rate.
        """
        self._controller.learning_rate = lr
        try:
            self._controller.fit(n_sgd_iters=max_iterations)
        except (NanError, RuntimeError):
            pass

        try:
            return self._controller.metrics_tracker[-1]["loss"]
        except IndexError:
            return np.inf
