# © Crown Copyright GCHQ
#
# Licensed under the GNU General Public License, version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Wrapping functions for use in Vanguard decorators.

Applying the :func:`wraps_class` decorator to a class will
update all method names and docstrings with those of the super class. The
:func:`process_args` function is a helper function for organising arguments
to a function into a dictionary for straightforward access.
"""

import inspect
import types
from functools import WRAPPER_ASSIGNMENTS, wraps
from typing import Any, Callable, Type, TypeVar

T = TypeVar("T")


def process_args(func: Callable, *args: Any, **kwargs: Any) -> dict:
    """
    Process the arguments for a function.

    Similar to :func:`inspect.getcallargs`, except it
    will repeatedly follow the ``__wrapped__`` attribute to
    get the correct function.  If func is passed as a bound
    function, then it will be converted into a bound function
    before :func:`inspect.getcallargs` is called.

    :param func: The function for which to process the arguments.
    :param args: Arguments to be passed to the function. Must be passed as args,
                        i.e. ``process_args(func, 1, 2)``.
    :param kwargs: Keyword arguments to be passed to the function. Must be passed as kwargs,
                            i.e. ``process_args(func, c=1)``.

    :returns: A mapping of parameter name to value for all parameters (including default ones) of the function.

    :Example:
        >>> def f(a, b, c=3, **kwargs):
        ...     pass
        >>>
        >>> process_args(f, 1, 2)
        {'a': 1, 'b': 2, 'c': 3}
        >>> process_args(f, a=1, b=2, c=4)
        {'a': 1, 'b': 2, 'c': 4}
        >>> process_args(f, a=1, b=2, c=4, e=5)
        {'a': 1, 'b': 2, 'c': 4, 'e': 5}
        >>> process_args(f, *(1,), **{'b': 2, 'c': 4})
        {'a': 1, 'b': 2, 'c': 4}
        >>> process_args(f, 1)
        Traceback (most recent call last):
        ...
        TypeError: f() missing 1 required positional argument: 'b'
    """
    func_self = getattr(func, "__self__", None)

    while True:
        try:
            func = func.__wrapped__
        except AttributeError:
            break

    try:
        func = types.MethodType(func, func_self)
    except TypeError:
        pass

    # TODO: This function is deprecated since python 3.5 - replace with inspect.Signature.bind() asap and remove
    #  this Pylint disable!
    # https://github.com/gchq/Vanguard/issues/203
    # pylint: disable=deprecated-method
    parameters_as_kwargs = inspect.getcallargs(func, *args, **kwargs)
    inner_kwargs = parameters_as_kwargs.pop("kwargs", {})
    parameters_as_kwargs.update(inner_kwargs)

    return parameters_as_kwargs


def wraps_class(base_class: Type[T]) -> Callable[[Type[T]], Type[T]]:
    r"""
    Update the names and docstrings of an inner class to those of a base class.

    This decorator controls the wrapping of an inner class, ensuring that all
    methods of the final class maintain the same names and docstrings as the
    inner class. Very similar to :func:`functools.wraps`.

    .. note::
        This decorator will return a class which seems almost identical to the
        base class, but a ``__wrapped__`` attribute will be added to point to the
        original class. All methods will be wrapped using :func:`functools.wraps`.

    :Example:
        >>> import inspect
        >>>
        >>> class First:
        ...     '''This is the first class.'''
        ...     def __init__(self, a, b):
        ...         pass
        >>>
        >>> @wraps_class(First)
        ... class Second(First):
        ...     '''This is the second class.'''
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        >>>
        >>> Second.__name__
        'First'
        >>> Second.__doc__
        'This is the first class.'
        >>> str(inspect.signature(Second.__init__))
        '(self, a, b)'
        >>> Second.__wrapped__
        <class 'vanguard.decoratorutils.wrapping.First'>
    """

    def inner_function(inner_class: Type[T]) -> Type[T]:
        """Update the values in the inner class."""
        for attribute in WRAPPER_ASSIGNMENTS:
            try:
                base_attribute_value = getattr(base_class, attribute)
            except AttributeError:
                pass
            else:
                setattr(inner_class, attribute, base_attribute_value)

        for key, value in inner_class.__dict__.items():
            if inspect.isfunction(value):
                try:
                    base_class_method = getattr(base_class, key)
                except AttributeError:
                    continue
                wrapped_method = wraps(base_class_method)(value)
                setattr(inner_class, key, wrapped_method)

        inner_class.__wrapped__ = base_class
        return inner_class

    return inner_function
