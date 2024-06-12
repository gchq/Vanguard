"""
Errors and warnings corresponding to unstable decorator combinations.

If a decorated class has implemented new functions (or overwritten existing ones)
then calling :meth:`~vanguard.decoratorutils.basedecorator.Decorator.verify_decorated_class`
will raise one of these errors or warnings.
"""


class DecoratorError(RuntimeError):
    """Base class for all decorator errors."""


class OverwrittenMethodError(DecoratorError):
    """An existing method has been overwritten."""


class UnexpectedMethodError(DecoratorError):
    """A new, unexpected method has been implemented."""


class TopmostDecoratorError(TypeError):
    """Attempting to decorate a top-level decorator."""


class MissingRequirementsError(ValueError):
    """Missing decorator requirements."""


class DecoratorWarning(RuntimeWarning):
    """Base class for all decorator warnings."""


class OverwrittenMethodWarning(DecoratorWarning):
    """An existing method has been overwritten."""


class UnexpectedMethodWarning(DecoratorWarning):
    """A new, unexpected method has been implemented."""
