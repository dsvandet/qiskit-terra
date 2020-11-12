# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Functions for working with Array dispatch."""

from typing import Union, Tuple

from .array import Array
from .dispatch import Dispatch


def wrap(func: callable, wrap_return: bool = False) -> callable:
    """Wrap an array backend function to work with Arrays.

    Args:
        func: a function to wrap.
        wrap_return: Optional. If true convert results that are
                     registered array backend types into Array objects.

    Returns:
        callable: The wrapped function.
    """

    def wrapped_function(*args, **kwargs):

        # Recursive wrap function arguments
        args = tuple(_wrap_function(x) if isinstance(x, type(lambda: 1))
                     else x for x in args)
        kwargs = dict((key, _wrap_function(val))
                      if isinstance(val, type(lambda: 1))
                      else (key, val) for key, val in kwargs.items())

        # Evaluate unwrapped function
        result = _wrap_function(func)(*args, **kwargs)

        # Optional wrap array return types back to Arrays
        if wrap_return:
            result = _wrap_return(result)
        return result

    return wrapped_function


def _wrap_return(result: Union[any, Tuple[any]]) -> Union[any, Tuple[any]]:
    """Wrap return array backend objects as Array objects"""
    if isinstance(result, tuple):
        result = tuple(Array(x)
                       if isinstance(x, Dispatch.REGISTERED_TYPES)
                       else x for x in result)
    elif isinstance(result, Dispatch.REGISTERED_TYPES):
        result = Array(result)
    return result


def _wrap_function(func: callable) -> callable:
    """Wrap a function to handle Array-like inputs and returns"""

    def wrapped_function(*args, **kwargs):

        # Unwrap inputs
        args = tuple(x.__qiskit_array__().data
                     if hasattr(x, '__qiskit_array__')
                     else x for x in args)
        kwargs = dict((key, val.__qiskit_array__().data)
                      if hasattr(val, '__qiskit_array__') else (key, val)
                      for key, val in kwargs.items())

        # Evaluate function with unwrapped inputs
        result = func(*args, **kwargs)

        # Unwrap result
        if isinstance(result, tuple):
            result = tuple(x.__qiskit_array__().data
                           if hasattr(x, '__qiskit_array__') else x
                           for x in result)
        elif hasattr(result, '__qiskit_array__'):
            result = result.__qiskit_array__().data
        return result

    return wrapped_function
