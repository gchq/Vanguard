"""
Contains any warnings that are needed in one place.
"""
import re

from gpytorch.utils.errors import NotPSDError
from gpytorch.utils.warnings import GPInputWarning, NumericalWarning

_INPUT_WARNING = "The input matches the stored training data. Did you forget to call model.train()?"
_CHOLESKY_WARNING = r"Runtime Error when computing Cholesky decomposition: (.*?)\. Using RootDecomposition\."
_JITTER_WARNING = r"A not p\.d\., added jitter of (.*?) to the diagonal"
_RE_INCORRECT_LIKELIHOOD_PARAMETER = re.compile(r"^.*?\.?__init__\(\) got (?:an unexpected|multiple values for) "
                                                r"keyword argument '(.*?)'$")

# This is so that pre-commits don't fail on unused imports.
__all__ = ["GPInputWarning", "NumericalWarning", "_INPUT_WARNING", "_CHOLESKY_WARNING", "_JITTER_WARNING",
           "_RE_INCORRECT_LIKELIHOOD_PARAMETER", "NotPSDError"]
