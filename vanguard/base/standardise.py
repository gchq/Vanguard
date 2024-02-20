"""
Contains a class decorator to apply input standard scaling to means and kernels.
"""
import torch
from numpy.typing import ArrayLike, DTypeLike

from ..decoratorutils import wraps_class


class StandardiseXModule:
    """
    A simple decorator to standard scale the inputs to a mean or kernel module before applying the mean or kernel.
    """
    def __init__(
            self,
            mean: ArrayLike[float],
            scale: ArrayLike[float],
            device: torch.device | None,
            dtype: DTypeLike | None,
    ) -> None:
        """
        Initialise self.

        :param mean: The mean (i.e. additive shift) of the standard scaling.
                Can be an array in the case of multiple features.
        :param scale: The scale (i.e. standard deviation) of the standard scaling.
                Can be an array in the case of multiple features.
        :param torch.device,None device: The device on which the mean and scale parameters should live.
        """
        self.mean = torch.as_tensor(mean, device=device, dtype=dtype)
        self.scale = torch.as_tensor(scale, device=device, dtype=dtype)

    def apply(
            self,
            module_class: torch.nn.Module,
    ) -> torch.nn.Module:
        """
        Modify the module's forward method to include standard scaling.

        :param module_class: The mean or kernel class to standard scale.
        :returns: The modified module class.
        """
        mean, scale = self.mean, self.scale

        @wraps_class(module_class)
        class ScaledModule(module_class):
            """An inner class which scales the forward method."""
            def forward(self, *args, **kwargs):
                """Scale the inputs before being passed."""
                scaled_args = ((arg - mean) / scale for arg in args)
                return super().forward(*scaled_args, **kwargs)

        return ScaledModule

    @classmethod
    def from_data(
            cls,
            x: torch.Tensor,
            device: torch.device | None,
            dtype: DTypeLike | None,
    ):
        """
        Create an instance of self with the mean and scale of the standard scaling obtained from the given data.

        :param x: (n_sample, n_features) The input data on which to learn to mean and scale.
        :param device: Where the mean and scale will reside.
        """
        mean, scale = x.mean(dim=0), x.std(dim=0)
        return cls(mean, scale, device, dtype)
