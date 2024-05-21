"""
Contains a spectral decomposition version of multivariate normal.
"""

from typing import NoReturn, Type, TypeVar

import torch
from torch.distributions import MultivariateNormal, constraints
from torch.distributions.utils import lazy_property

T = TypeVar("T", bound="MultivariateNormal")


class SpectralRegularisedMultivariateNormal(MultivariateNormal):
    r"""
    Construct a multivariate normal distribution from a spectral decomposition of its covariance matrix.

    The covariance matrix is defined by its eigenvectors and eigenvalues.
    The class itself is just the torch MultivariateNormal class, but with a new constructor method,
    and minimum necessary modifications to make it compatible.

    .. note::
        We are abusing the lower-triangular Cholesky factorisation matrix of the standard
        MultivariateNormal class. As such, we have to disable the lower-triangular constraint.
        Otherwise, the only method that becomes invalid is `precision_matrix`, which relies
        on the lower triangular form to do efficient matrix inversion. It is of course possible
        to compute the precision matrix here, but it is not needed and will generally be numerically
        unstable, so it is just disabled.
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
    }

    @lazy_property
    def precision_matrix(self) -> NoReturn:
        raise NotImplementedError("Precision is not available for spectral defined multivariate normals.")

    @classmethod
    def from_eigendecomposition(
        cls: Type[T], mean: torch.Tensor, covar_eigenvalues: torch.Tensor, covar_eigenvectors: torch.Tensor
    ) -> Type[T]:
        """
        Construct the distribution from the eigendecomposition of its covariance matrix.

        :param mean: Mean of the multivariate normal.
        :param covar_eigenvalues: The eigenvalues of the covariance matrix.
        :param covar_eigenvectors: The eigenvectors of the covariance matrix,
                                                (columns are the eigenvectors).
        """
        tril = torch.einsum("...ij,...jk->...ik", covar_eigenvectors, torch.diag_embed(covar_eigenvalues.sqrt()))
        return cls(loc=mean, scale_tril=tril)
