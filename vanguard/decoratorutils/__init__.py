"""
Tools to support decorators in Vanguard.

In Vanguard, decorators allow for easy, dynamic subclassing of
:py:class:`~vanguard.base.gpcontroller.GPController` instances, to add new functionality
in an easily composable way. All new decorators should subclass from
:py:class:`~basedecorator.Decorator` or :py:class:`~basedecorator.TopMostDecorator`.
See :doc:`../examples/decorator_walkthrough` for more details.
"""
from .basedecorator import Decorator, TopMostDecorator
from .wrapping import process_args, wraps_class
