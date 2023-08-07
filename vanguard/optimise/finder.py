"""
Contains the LearningRateFinder class to aid with choosing the largest possible learning rate.
"""
from gpytorch.utils.errors import NanError
import matplotlib.pyplot as plt
import numpy as np


class LearningRateFinder:
    """
    Estimates the best learning rate for a controller/data combination.

    Try an increasing geometric sequence of learning rates for a small number of iterations to find
    the best learning rate (i.e. the largest learning rate giving stable training).
    """
    def __init__(self, controller):
        """
        Initialise self.

        :param vanguard.base.gpcontroller.GPController controller: An instantiated vanguard GP controller
                                                                    whose learning rate shall be optimised.
        """
        self._controller = controller
        self._learning_rates = []
        self._losses = []

    @property
    def best_learning_rate(self):
        return self._learning_rates[np.argmin(self._losses)]

    def find(self, start_lr=1e-5, end_lr=10, num_divisions=100, max_iterations=20):
        """
        Try the range of learning rates and record the loss obtained.

        :param float start_lr: The smallest learning rate to try.
        :param float end_lr: The largest learning rate to try.
        :param int num_divisions: The number of learning rates to try.
        :param int max_iterations: The top number of iterations of gradient descent to run for each learning rate.
        """
        ratio = np.power(end_lr/start_lr, 1./num_divisions)
        self._learning_rates = [start_lr * ratio ** index for index in range(num_divisions)]
        self._losses = [self._run_learning_rate(lr, max_iterations) for lr in self._learning_rates]

    def plot(self, **kwargs):
        """
        Plot the obtained loss-vs-lr curve.
        """
        plt.plot(self._learning_rates, self._losses, **kwargs)
        plt.xlabel("learning rate")
        plt.ylabel("best loss")
        plt.show()

    def _run_learning_rate(self, lr, max_iterations):
        """
        Do the training for a single learning rate.

        If training fails due to NaNs before the max iterations is reached,
        the loss is taken to be infinity.

        :param float lr: The learning rate.
        :param int max_iterations: The maximum number of gradient descent iterations to run.
        :returns: The loss at the end of training with the given learning rate.
        :rtype: float
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
