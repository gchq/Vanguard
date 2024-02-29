"""
Enable training on non-Gaussian observation noise with warping.

Warp functions are used to map data to a different domain to train the Gaussian
process. Although Vanguard contains many pre-written warp functions, any new
ones can be created by subclassing class:`~basefunction.WarpFunction` and
implementing the meth:`~basefunction.WarpFunction.forward`,
meth:`~basefunction.WarpFunction.inverse` and (optionally)
meth:`~basefunction.WarpFunction.deriv` methods.

Warp functions are applied to a class:`~vanguard.base.gpcontroller.GPController`
subclass using the class:`SetWarp` decorator.
"""
from .basefunction import MultitaskWarpFunction, WarpFunction
from .decorator import SetWarp
from .distribution import WarpedGaussian
from .input import SetInputWarp
from .intermediate import require_controller_input
