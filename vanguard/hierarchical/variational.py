"""
Contains the VariationalHierarchicalHyperparameters decorator.
"""
import torch
from gpytorch.lazy import lazify
from gpytorch.variational import CholeskyVariationalDistribution

from ..decoratorutils import wraps_class
from .base import (BaseHierarchicalHyperparameters,
                   extract_bayesian_hyperparameters, set_batch_shape)
from .collection import HyperparameterCollection


class VariationalHierarchicalHyperparameters(BaseHierarchicalHyperparameters):
    """
    Convert a controller so that variational inference is performed over its hyperparameters.

    Note that only those hyperparameters specified using the
    :class:`~vanguard.hierarchical.module.BayesianHyperparameters` decorator will be included
    for variational inference. The remaining hyperparameters will be inferred as point estimates.

    :Example:
        >>> from gpytorch.kernels import RBFKernel
        >>> import numpy as np
        >>> import torch
        >>> from vanguard.vanilla import GaussianGPController
        >>> from vanguard.hierarchical import (BayesianHyperparameters,
        ...                                    VariationalHierarchicalHyperparameters)
        >>>
        >>> @VariationalHierarchicalHyperparameters(num_mc_samples=50)
        ... class HierarchicalController(GaussianGPController):
        ...     pass
        >>>
        >>> @BayesianHyperparameters()
        ... class BayesianRBFKernel(RBFKernel):
        ...     pass
        >>>
        >>> train_x = np.array([0, 0.5, 0.9, 1])
        >>> train_y = 1 / (1 + train_x) + np.random.randn(4)*0.005
        >>> gp = HierarchicalController(train_x, train_y, BayesianRBFKernel, y_std=0)
        >>> loss = gp.fit(100)
        >>>
        >>> test_x = np.array([0.05, 0.95])
        >>> mean, lower, upper = gp.posterior_over_point(test_x).confidence_interval()
        >>> (upper > 1/(1 + test_x)).all(), (lower < 1/(1 + test_x)).all()
        (True, True)
    """
    def __init__(self, num_mc_samples=100, variational_distribution_class=CholeskyVariationalDistribution, **kwargs):
        """
        Initialise self.

        :param int num_mc_samples: The number of Monte Carlo samples to use when approximating
                                    intractable integrals in the variational ELBO and the
                                    predictive posterior.
        :param gpytorch.variational._VariationalDistribution,None variational_distribution_class:
            The variational distribution to use for the raw hyperparameters' posterior. Defaults
            to :class:`~gpytorch.variational.CholeskyVariationalDistribution`.
        """
        super().__init__(num_mc_samples=num_mc_samples, **kwargs)
        self.variational_distribution_class = variational_distribution_class

    def _decorate_class(self, cls):
        sample_shape = self.sample_shape
        variational_distribution_class = self.variational_distribution_class
        base_decorated_cls = super()._decorate_class(cls)

        @wraps_class(base_decorated_cls)
        class InnerClass(base_decorated_cls):
            def __init__(self, *args, **kwargs):
                for module_name in ("kernel", "mean", "likelihood"):
                    set_batch_shape(kwargs, module_name, sample_shape)

                super().__init__(*args, **kwargs)

                module_hyperparameter_pairs, point_estimate_kernels = extract_bayesian_hyperparameters(self)
                _correct_point_estimate_shapes(point_estimate_kernels)

                self.hyperparameter_collection = HyperparameterCollection(module_hyperparameter_pairs, sample_shape,
                                                                          variational_distribution_class)

                self._smart_optimiser.update_registered_module(self._gp)
                self._smart_optimiser.register_module(self.hyperparameter_collection.variational_distribution)

            def _loss(self, train_x, train_y):
                """Add KL term to loss and average over hyperparameter samples."""
                self.hyperparameter_collection.sample_and_update()
                nmll = super()._loss(train_x, train_y)
                return nmll.mean() + self.hyperparameter_collection.kl_term()

        return InnerClass

    @staticmethod
    def _infinite_posterior_samples(controller, x):
        """
        Yield posterior samples forever.

        :param vanguard.base.gpcontroller.GPController controller: The controller from which to yield samples.
        :param array_like[float] x: (n_preds, n_features) The predictive inputs.
        """
        tx = torch.as_tensor(x, dtype=torch.float32, device=controller.device)
        while True:
            controller.hyperparameter_collection.sample_and_update()
            output = _safe_index_batched_multivariate_normal(controller._gp_forward(tx).add_jitter(1e-3))
            yield from output

    @staticmethod
    def _infinite_fuzzy_posterior_samples(controller, x, x_std):
        """
        Yield fuzzy posterior samples forever.

        :param vanguard.base.gpcontroller.GPController controller: The controller from which to yield samples.
        :param array_like[float] x: (n_preds, n_features) The predictive inputs.
        :param array_like[float],float x_std: The input noise standard deviations:

            * array_like[float]: (n_features,) The standard deviation per input dimension for the predictions,
            * float: Assume homoskedastic noise.
        """
        tx = torch.tensor(x, dtype=torch.float32, device=controller.device)
        tx_std = controller._process_x_std(x_std).to(controller.device)
        while True:
            controller.hyperparameter_collection.sample_and_update()
            # This cunning trick matches the sampled x shape to the MC samples batch shape.
            # The results is that each sample from output comes from independent x samples
            # and from independent variational posterior samples.
            sample_shape = controller.hyperparameter_collection.sample_shape + tx.shape
            x_sample = torch.randn(size=sample_shape, device=controller.device)*tx_std + tx
            output = _safe_index_batched_multivariate_normal(controller._gp_forward(x_sample).add_jitter(1e-3))
            yield from output

    @staticmethod
    def _infinite_likelihood_samples(controller, x):
        """
        Yield likelihood samples forever.

        :param vanguard.base.gpcontroller.GPController controller: The controller from which to yield samples.
        :param array_like[float] x: (n_preds, n_features) The predictive inputs.
        """
        tx = torch.as_tensor(x, dtype=torch.float32, device=controller.device)
        while True:
            controller.hyperparameter_collection.sample_and_update()
            output = _safe_index_batched_multivariate_normal(controller._gp_forward(tx).add_jitter(1e-3))
            for sample in output:
                shape = controller._decide_noise_shape(controller.posterior_class(sample), x)
                noise = torch.zeros(shape, dtype=torch.float32, device=controller.device)
                likelihood_output = controller._likelihood(sample, noise=noise)
                yield likelihood_output

    @staticmethod
    def _infinite_fuzzy_likelihood_samples(controller, x, x_std):
        """
        Yield fuzzy likelihood samples forever.

        :param vanguard.base.gpcontroller.GPController controller: The controller from which to yield samples.
        :param array_like[float] x: (n_preds, n_features) The predictive inputs.
        :param array_like[float],float x_std: The input noise standard deviations:

            * array_like[float]: (n_features,) The standard deviation per input dimension for the predictions,
            * float: Assume homoskedastic noise.
        """
        tx = torch.tensor(x, dtype=torch.float32, device=controller.device)
        tx_std = controller._process_x_std(x_std).to(controller.device)

        while True:
            controller.hyperparameter_collection.sample_and_update()
            # This cunning trick matches the sampled x shape to the MC samples batch shape.
            # The results is that each sample from output comes from independent x samples
            # and from independent variational posterior samples.
            sample_shape = controller.hyperparameter_collection.sample_shape + tx.shape
            x_sample = torch.randn(size=sample_shape, device=controller.device)*tx_std + tx
            output = _safe_index_batched_multivariate_normal(controller._gp_forward(x_sample).add_jitter(1e-3))
            for sample in output:
                shape = controller._decide_noise_shape(controller.posterior_class(sample), x)
                noise = torch.zeros(shape, dtype=torch.float32, device=controller.device)
                likelihood_output = controller._likelihood(sample, noise=noise)
                yield likelihood_output


def _correct_point_estimate_shapes(point_estimate_kernels):
    """
    Adjust the shape of the constants of point estimate kernels.

    These will be incorrect due to how GPyTorch handles batch shapes.
    """
    for point_estimate_scale_kernel in point_estimate_kernels:
        delattr(point_estimate_scale_kernel, "raw_outputscale")
        point_estimate_scale_kernel.register_parameter(name="raw_outputscale",
                                                       parameter=torch.nn.Parameter(torch.zeros([1, ])))


def _safe_index_batched_multivariate_normal(distribution):
    """
    Delazifies the batched covariance matrix and yields recreated non-batch normals.

    Indexing into the batch dimension of batch :class:`~gpytorch.distributions.MultivariateNormal`
    is somewhat brittle when the underlying covariance matrix is lazy (which happens when the covariance
    matrix is larger than an obscure threshold). Hopefully this will change, but for now, we will work
    around it. This function delazifies the batched covariance matrix and yields recreated non-batch
    normals using then relazified individual covariance matrices.
    Delazifiying the batch covariance matrix doesn't cause any inefficiencies because the individual
    covariance matrices would be delazified later anyway. Relazifying the individual matrices just
    delays any Choleksy issues, which is good because we have handling for them downstream.
    """
    distribution_type = type(distribution)
    non_lazy_covariance_matrix = distribution.covariance_matrix
    for sub_mean, sub_covariance_matrix in zip(distribution.mean, non_lazy_covariance_matrix):
        yield distribution_type(sub_mean, lazify(sub_covariance_matrix))
