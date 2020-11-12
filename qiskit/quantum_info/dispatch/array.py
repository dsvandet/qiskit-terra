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
"""Array Class"""

from typing import Optional
from numbers import Number

import numpy
from numpy.lib.mixins import NDArrayOperatorsMixin
from .dispatch import Dispatch, asarray

__all__ = ['Array']


class Array(NDArrayOperatorsMixin):
    """Qiskit Array class.

    This class provides a Numpy compatible wrapper to supported Python
    array libraries. Supported backends are 'numpy' and 'jax'.

    **Numpy compatibility**

    This class is compatible with Numpy functions and will handle dispatching
    to the correct corresponding function for the current Array backend if
    one is supported.

    **Attributes and Methods**

    This class implements two custom attributes
        * :attr`data` which returns the wrapped array object
        * :attr`backend` which returns the backend string of hte wrapped array.

    All other attributes and methods of the wrapped array are accessable
    through this class, however their return type will be converted to an
    Array object rather.
    """
    def __init__(self,
                 data: any,
                 dtype: Optional[any] = None,
                 order: Optional[str] = None,
                 backend: Optional[str] = None):
        """Initialize an Array container.

        Args:
            data: An array_like input. This can be an object of any type
                  supported by the registered `asarray` method for the
                  specified backend.
            dtype: Optional. The dtype of the returned array. This value
                   must be supported by the specified array backend.
            order: Optional. The array order. This value must be supported
                   by the specified array backend.
            backend: A registered array backend name. If None the
                     default array backend will be used.

        Raises:
            ValueError: if input cannot be converted to an Array.
        """
        if hasattr(data, '__qiskit_array__'):
            array = data.__qiskit_array__()
            if not isinstance(array, Array):
                raise ValueError('object __qiskit_array__ method is not producing an Array')
            self._data = array._data
            self._backend = array._backend
        else:
            self._data = asarray(data,
                                 dtype=dtype,
                                 order=order,
                                 backend=backend)
            self._backend = backend if backend else Dispatch.backend(self._data)

    @property
    def data(self):
        """Return the wrapped array data object"""
        return self._data

    @data.setter
    def data(self, value):
        """Update the wrapped array data object"""
        self._data[:] = value

    @property
    def backend(self):
        """Return the backend of the wrapped array class"""
        return self._backend

    @backend.setter
    def backend(self, value: str):
        """Set the backend of the wrapped array class"""
        Dispatch.validate_backend(value)
        self._data = asarray(self._data, backend=value)
        self._backend = value

    def __repr__(self):
        prefix = 'Array('
        if self._backend == Dispatch.DEFAULT_BACKEND:
            suffix = ")"
        else:
            suffix = "backend='{}')".format(self._backend)
        return Dispatch.repr(self.backend)(
            self._data, prefix=prefix, suffix=suffix)

    def __getitem__(self, key: str) -> any:
        """Return value from wrapped array"""
        return self._data[key]

    def __setitem__(self, key: str, value: any):
        """Return value of wrapped array"""
        self._data[key] = value

    def __setattr__(self, name: str, value: any):
        """Set attribute of wrapped array."""
        if name in ('_data', 'data', '_backend', 'backend'):
            super().__setattr__(name, value)
        else:
            setattr(self._data, name, value)

    def __getattr__(self, name: str) -> any:
        """Get attribute of wrapped array and convert to an Array."""
        # Call attribute on inner array object
        result = getattr(self._data, name)

        # If return object is an Array backend type wrap result
        # back into an Array
        if isinstance(result, tuple):
            tmp = tuple()
            for i in result:
                backend = Dispatch.backend(i)
                tmp += (Array(i, backend=backend) if backend else i, )
            result = tmp
        else:
            backend = Dispatch.backend(result)
            if backend:
                result = Array(result, backend=backend)
        return result

    def __qiskit_array__(self):
        return self

    def __array__(self) -> numpy.ndarray:
        if isinstance(self._data, numpy.ndarray):
            return self._data
        return numpy.asarray(self._data)

    def __str__(self) -> str:
        return str(self._data)

    def __int__(self):
        """Convert size 1 array to an int."""
        if numpy.size(self) != 1:
            raise TypeError('only size-1 Arrays can be converted to Python scalars')
        return int(self._data)

    def __float__(self):
        """Convert size 1 array to a float."""
        if numpy.size(self) != 1:
            raise TypeError('only size-1 Arrays can be converted to Python scalars')
        return float(self._data)

    def __complex__(self):
        """Convert size 1 array to a complex."""
        if numpy.size(self) != 1:
            raise TypeError('only size-1 Arrays can be converted to Python scalars')
        return complex(self._data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Dispatcher for numpy ufuncs to support the wrapped array backend."""
        if method != '__call__':
            return NotImplemented

        out = kwargs.get('out', ())

        for x in inputs + out:
            # Only support operations with instances of REGISTERED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, Dispatch.REGISTERED_TYPES +
                              (Array, Number)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(i._data if isinstance(i, Array) else i for i in inputs)
        if out:
            kwargs['out'] = tuple(i._data if isinstance(i, Array) else i
                                  for i in out)

        # Get implementation for backend
        backend = self.backend
        dispatch_func = Dispatch.array_ufunc(backend, ufunc, method)
        if dispatch_func == NotImplemented:
            return NotImplemented

        result = dispatch_func(*inputs, **kwargs)

        # Format Results
        if isinstance(result, tuple):
            # multiple return values
            return tuple(Array(i, backend=backend) for i in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return Array(result, backend=backend)

    def __array_function__(self, func, types, args, kwargs):
        """Dispatcher for numpy array function to support the wrapped array backend."""
        if not all(issubclass(t, (Array,) + Dispatch.REGISTERED_TYPES) for t in types):
            return NotImplemented

        args = tuple(i._data if isinstance(i, Array) else i for i in args)
        out = kwargs.get('out', ())
        if out:
            kwargs['out'] = tuple(i._data if isinstance(i, Array) else i
                                  for i in out)

        # Get implementation for backend
        backend = self.backend
        dispatch_func = Dispatch.array_function(backend, func)
        if dispatch_func == NotImplemented:
            return NotImplemented
        result = dispatch_func(*args, **kwargs)

        # Format Results
        if isinstance(result, tuple):
            # multiple return values
            return tuple(
                Array(i, backend=backend
                      ) if isinstance(i, Dispatch.REGISTERED_TYPES) else i
                for i in result)
        elif isinstance(result, Dispatch.REGISTERED_TYPES):
            return Array(result, backend=backend)
        else:
            return result
