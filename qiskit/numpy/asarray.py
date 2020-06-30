# -*- coding: utf-8 -*-

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

"""
Tensor backend math functions
"""

from functools import singledispatch

from numpy import ndarray
from numpy import asarray as asarray_numpy

from qiskit.exceptions import QiskitError

from .numpy_jax import _HAS_JAX, JaxArray


@singledispatch
def array_backend(array):
    """Return the backend string of an array object."""
    # pylint: disable=unused-argument
    return None


@array_backend.register(ndarray)
def _(array):
    return 'numpy'


if _HAS_JAX:
    from jax.numpy import asarray as asarray_jax
    _JAX_TYPE = [JaxArray]

    @array_backend.register(JaxArray)
    def _(array):
        return 'jax'

    @singledispatch
    def _to_jax(tensor, dtype=None):
        # There is a bug with JAX as array that will not convert
        # an input Numpy array to a JAX DeviceArray if the specified
        # dtype matches the array dtype.
        if dtype is None or dtype == tensor.dtype:
            return asarray_jax(tensor)
        return asarray_jax(tensor, dtype=dtype)

    @_to_jax.register(JaxArray)
    def _(tensor, dtype=None):
        if dtype is None or dtype == tensor.dtype:
            return tensor
        return asarray_jax(tensor, dtype=dtype)
else:
    _JAX_TYPE = []


# Currently support backends if the Python respective packages are installed
_ARRAY_TYPES = tuple([ndarray] + _JAX_TYPE + [list])
_SUPPORTED_BACKENDS = ['numpy', 'jax']


def array_backends():
    """Return Tensor backends that are installed on this system"""
    backends = ['numpy']
    if _HAS_JAX:
        backends.append('jax')
    return backends


def is_array(tensor):
    """Check if tensor is an array"""
    return isinstance(tensor, _ARRAY_TYPES)


def asarray(tensor, dtype=None, backend=None):
    """Convert input to a tensor on the specified backend."""
    if backend == 'numpy' or (
            backend is None and isinstance(tensor, (ndarray, list))):
        return asarray_numpy(tensor, dtype=dtype)

    if _HAS_JAX and (backend == 'jax' or (
            backend is None and isinstance(tensor, JaxArray))):
        return _to_jax(tensor, dtype=dtype)

    if backend in _SUPPORTED_BACKENDS:
        raise QiskitError(
            '{} array backend is not available on this system.'
            ' Available backends are {}'.format(backend, array_backends()))
    return QiskitError('Invalid Tensor backend: {}'.format(backend))
