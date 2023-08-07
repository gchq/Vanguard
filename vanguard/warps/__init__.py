"""
Enable training on non-Gaussian observation noise with warping.

Warp functions are used to map data to a different domain to train the Gaussian
process. Although Vanguard contains many pre-written warp functions, any new
ones can be created by subclassing :py:class:`~basefunction.WarpFunction` and
implementing the :py:meth:`~basefunction.WarpFunction.forward`,
:py:meth:`~basefunction.WarpFunction.inverse` and (optionally)
:py:meth:`~basefunction.WarpFunction.deriv` methods.

Warp functions are applied to a :py:class:`~vanguard.base.gpcontroller.GPController`
subclass using the :py:class:`SetWarp` decorator.
"""
from .basefunction import MultitaskWarpFunction, WarpFunction
from .decorator import SetWarp
from .distribution import WarpedGaussian
from .input import SetInputWarp
from .intermediate import require_controller_input
