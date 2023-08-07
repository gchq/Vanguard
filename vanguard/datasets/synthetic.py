"""
Synthetic data is particularly useful when running tests, as the data can be specifically cultivated for one's needs.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

from .basedataset import Dataset


def simple_f(x):
    r"""
    Map values through a simple equation.

    .. math::
        f(x) = \sin(2\pi x)
    """
    return np.sin(2 * np.pi * x)


def complicated_f(x):
    r"""
    Map values through a complicated equation.

    .. math::
        f(x) = -x^\frac{3}{2} + x\sin^2(2\pi x)
    """
    return -x ** 3 / 2 + x * simple_f(x) ** 2


def very_complicated_f(x):
    r"""
    Map values through a *very* complicated equation.

    .. math::
        f(x) = -x^\frac{3}{2} + x\sin^2(2\pi x) + x^2 \cos(10\pi x)
    """
    return complicated_f(x) + x ** 2 * np.cos(10 * np.pi * x)


class SyntheticDataset(Dataset):
    """
    Synthetic data with homoskedastic noise for testing.
    """
    def __init__(self, functions=(simple_f,), output_noise=0.1, train_input_noise_bounds=(0.01, 0.05),
                 test_input_noise_bounds=(0.01, 0.03), n_train_points=30, n_test_points=50, significance=0.025):
        """
        Initialise self.

        :param Iterable[function] functions: The functions to be used to generate the synthetic data.
        :param float output_noise: The standard deviation for the output standard deviation, defaults to 0.1.
        :param train_input_noise_bounds: The lower, upper bounds of the linearly varying noise
            for the training input. Defaults to (0.01, 0.05).
        :param test_input_noise_bounds: The lower, upper bounds of the linearly varying noise
            for the testing input. Defaults to (0.01, 0.03).
        :param int n_train_points: The total number of training points.
        :param int n_test_points: The total number of testing points.
        :param float significance: The significance to be used.
        """
        self.functions = list(functions)

        train_data = self.make_sample_data(n_train_points, train_input_noise_bounds, output_noise)
        test_data = self.make_sample_data(n_test_points, test_input_noise_bounds, 0)

        (train_x, train_x_std), train_y = train_data
        (test_x, test_x_std), test_y = test_data

        train_y_std = output_noise
        test_y_std = output_noise

        super().__init__(train_x, train_x_std, train_y, train_y_std,
                         test_x, test_x_std, test_y, test_y_std,
                         significance)

    def make_sample_data(self, n_points, input_noise_bounds, output_noise_level, interval_length=1):
        """
        Create some sample data.

        :param int n_points: The number of points to create.
        :param tuple[float] input_noise_bounds: The lower, upper bounds for the resulting input noise.
        :param float output_noise_level: The amount of noise applied to the inputs.
        :param float interval_length: Use to scale the exact image of the function, defaults to 1.
        :return: The output and the mean and standard deviation of the input, in the form ``(x_mean, x_std), y``.
        :rtype: tuple[tuple[numpy.ndarray[float], numpy.ndarray[float]]
        """
        x_mean = np.linspace(0, interval_length, n_points)
        x_std = np.linspace(*input_noise_bounds, n_points)

        individual_ys = []
        for function in self.functions:
            exact_y = function(x_mean * 2 / interval_length)
            y_noise = np.random.normal(size=x_mean.shape) * output_noise_level
            y = exact_y + y_noise
            individual_ys.append(y)

        y = np.squeeze(np.stack(individual_ys, axis=-1))

        scaler = StandardScaler()
        x_mean = scaler.fit_transform(x_mean.reshape(-1, 1))
        x_std = x_std.reshape(-1, 1) / scaler.scale_
        return (x_mean, x_std), y


class MultidimensionalSyntheticDataset(Dataset):
    """Synthetic data with multiple input dimensions."""
    def __init__(self, functions=(simple_f, complicated_f), **kwargs):
        """
        Initialise self.

        :param Iterable[function] functions: The functions used on each input dimension
                                                (they are combined linearly to make a single output).
        """
        one_dimensional_datasets = [SyntheticDataset(functions=(function,), **kwargs) for function in functions]
        train_x = np.stack([dataset.train_x.ravel() for dataset in one_dimensional_datasets], -1)
        train_x_std = np.stack([dataset.train_x_std.ravel() for dataset in one_dimensional_datasets], -1)
        train_y = np.mean(np.stack([dataset.train_y.ravel() for dataset in one_dimensional_datasets], -1), axis=-1)
        train_y_std = one_dimensional_datasets[0].train_y_std

        test_x = np.stack([dataset.test_x.ravel() for dataset in one_dimensional_datasets], -1)
        test_x_std = np.stack([dataset.test_x_std.ravel() for dataset in one_dimensional_datasets], -1)
        test_y = np.mean(np.stack([dataset.test_y.ravel() for dataset in one_dimensional_datasets], -1), axis=-1)
        test_y_std = one_dimensional_datasets[0].test_y_std
        super().__init__(train_x, train_x_std, train_y, train_y_std,
                         test_x, test_x_std, test_y, test_y_std,
                         kwargs.pop("significance", 0.025))


class HeteroskedasticSyntheticDataset(SyntheticDataset):
    """
    Synthetic data with heteroskedastic noise for testing.

    The ``train_y_std`` and ``test_y_std`` attributes are created by drawing from a normal distribution centred
    on the value of the ``output_noise`` parameter.
    """
    def __init__(self, functions=(simple_f,), output_noise=0.1, train_input_noise_bounds=(0.01, 0.05),
                 test_input_noise_bounds=(0.01, 0.03), n_train_points=30, n_test_points=50, significance=0.025):
        """
        Initialise self.

        :param Iterable[function] functions: The functions to be used to generate the synthetic data.
        :param float output_noise: The standard deviation for the output standard deviation, defaults to 0.1.
        :param train_input_noise_bounds: The lower, upper bounds of the linearly varying noise
            for the training input. Defaults to (0.01, 0.05).
        :param test_input_noise_bounds: The lower, upper bounds of the linearly varying noise
            for the testing input. Defaults to (0.01, 0.03).
        :param int n_train_points: The total number of training points.
        :param int n_test_points: The total number of testing points.
        :param float significance: The significance to be used.
        """
        super().__init__(functions, output_noise, train_input_noise_bounds, test_input_noise_bounds, n_train_points,
                         n_test_points, significance)
        self.train_y_std = (np.random.normal(loc=self.train_y_std, scale=0.01, size=n_train_points))
        self.test_y_std = (np.random.normal(loc=self.test_y_std, scale=0.01, size=n_train_points))


class HigherRankSyntheticDataset(Dataset):
    """
    Synthetic data with rank 2 input features. In this case each x is a 2x2 matrix.
    """
    def __init__(self, functions=(simple_f,), output_noise=0.1, train_input_noise_bounds=(0.01, 0.05),
                 test_input_noise_bounds=(0.01, 0.03), n_train_points=30, n_test_points=50, significance=0.025):
        """
        Initialise self.

        :param Iterable[function] functions: The functions to be used to generate the synthetic data.
        :param float output_noise: The standard deviation for the output standard deviation, defaults to 0.1.
        :param train_input_noise_bounds: The lower, upper bounds of the linearly varying noise
            for the training input. Defaults to (0.01, 0.05).
        :param test_input_noise_bounds: The lower, upper bounds of the linearly varying noise
            for the testing input. Defaults to (0.01, 0.03).
        :param int n_train_points: The total number of training points.
        :param int n_test_points: The total number of testing points.
        :param float significance: The significance to be used.
        """
        self.functions = list(functions)

        train_data = self.make_sample_data(
            n_train_points, train_input_noise_bounds, output_noise)
        test_data = self.make_sample_data(
            n_test_points, test_input_noise_bounds, 0)

        (train_x, train_x_std), train_y = train_data
        (test_x, test_x_std), test_y = test_data

        train_y_std = output_noise
        test_y_std = output_noise

        super().__init__(train_x, train_x_std, train_y, train_y_std,
                         test_x, test_x_std, test_y, test_y_std,
                         significance)

    def make_sample_data(self, n_points, input_noise_bounds, output_noise_level, interval_length=1):
        """
        Create some sample data.

        :param int n_points: The number of points to create.
        :param tuple[float] input_noise_bounds: The lower, upper bounds for the resulting input noise.
        :param float output_noise_level: The amount of noise applied to the inputs.
        :param float interval_length: Use to scale the exact image of the function, defaults to 1.
        :return: The output and the mean and standard deviation of the input, in the form ``(x_mean, x_std), y``.
        :rtype: tuple[tuple[numpy.ndarray[float], numpy.ndarray[float]]
        """
        x_mean = np.linspace(0, interval_length, n_points)
        x_mean = np.random.randn(n_points, 2, 2) + x_mean.reshape(-1, 1, 1)

        x_std = np.linspace(*input_noise_bounds, n_points)
        x_std = np.ones((n_points, 2, 2)) + x_std.reshape(-1, 1, 1)

        fixed_matrix = np.array([[1, 0.5], [0, 2]])

        individual_ys = []
        for function in self.functions:
            exact_y = function(x_mean @ fixed_matrix * 2 / interval_length).mean(axis=1).mean(axis=1)
            y_noise = np.random.normal(size=x_mean.shape[0]) * output_noise_level
            y = exact_y + y_noise
            individual_ys.append(y)

        y = np.squeeze(np.stack(individual_ys, axis=-1))

        return (x_mean, x_std), y
