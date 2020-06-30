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
import numpy as np
import scipy.linalg as la


# -------------------------------------------------------------------------
# Linear Algebra functions
# -------------------------------------------------------------------------

@singledispatch
def trace(array, offset=0, axis1=0, axis2=1, dtype=None):
    """Trace of an array"""
    return np.trace(array, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)


@singledispatch
def norm(array, ord=None, axis=None, keepdims=False):
    """Return the norm of an array"""
    return la.norm(array, ord=ord, axis=axis, keepdims=keepdims)


# -------------------------------------------------------------------------
# Array transformation functions
# -------------------------------------------------------------------------

@singledispatch
def dot(array, other):
    """Dot product of two arrays."""
    return np.dot(array, other)


@singledispatch
def kron(array, other):
    """Kronecker product of two arrays."""
    return np.kron(array, other)


@singledispatch
def outer(array, other):
    """Outer product of two arrays."""
    return np.outer(array, other)


@singledispatch
def transpose(array, axes=None):
    """Transpose a array on axes."""
    return np.transpose(array, axes=axes)


@singledispatch
def conjugate(array):
    """Complex conjugate of an array"""
    return conj(array)


@singledispatch
def conj(array):
    """Complex conjugate of an array"""
    return np.conj(array)


@singledispatch
def adjoint(array, *axes):
    """Complex conjugate transpose of an array"""
    return conj(transpose(array, *axes))


@singledispatch
def sum(array, axis=None, dtype=None, keepdims=False):
    """Sum a array along the specified axis."""
    return np.sum(array, axis=axis, dtype=dtype, keepdims=keepdims)


@singledispatch
def prod(array, axis=None, dtype=None, keepdims=False):
    """Multiply reduce a array along the specified axis."""
    return np.prod(array, axis=axis, dtype=dtype, keepdims=keepdims)


@singledispatch
def diag(arr, k=0):
    """Extract a diagonal or construct a diagonal array."""
    return np.diag(arr, k=k)


# -------------------------------------------------------------------------
# Array shape functions
# -------------------------------------------------------------------------

@singledispatch
def shape(array):
    """Return the shape of an array"""
    return np.shape(array)


@singledispatch
def reshape(array, new_shape, order='C'):
    """Reshape an array"""
    return np.reshape(array, new_shape, order=order)


@singledispatch
def flatten(array, order='C'):
    """Flatten an array"""
    return np.asarray(array).flatten(order=order)


@singledispatch
def ravel(array, order='C'):
    """Flatten an array"""
    return np.ravel(array, order=order)


# -------------------------------------------------------------------------
# Array elementwise functions
# -------------------------------------------------------------------------

@singledispatch
def real(array):
    """Return the real part of an array"""
    return np.real(array)


@singledispatch
def imag(array):
    """Return the imaginary part of an array"""
    return np.imag(array)


@singledispatch
def abs(array):
    """Return the elementwise absolute value of an array"""
    return np.abs(array)


@singledispatch
def round(array, decimals=0):
    """Round an array"""
    return np.round(array, decimals)


@singledispatch
def isclose(array, target, rtol=1e-5, atol=1e-8):
    """Sum a array along the specified axis."""
    return np.isclose(array, target, rtol=rtol, atol=atol)


@singledispatch
def allclose(array, target, rtol=1e-5, atol=1e-8):
    """Multiply reduce a array along the specified axis."""
    return np.allclose(array, target, rtol=rtol, atol=atol)


@singledispatch
def sqrt(array):
    """Elementwise sqrt of an array"""
    return np.sqrt(array)


@singledispatch
def sin(array):
    """Elementwise sin of an array"""
    return np.sin(array)


@singledispatch
def cos(array):
    """Elementwise cos of an array"""
    return np.cos(array)


@singledispatch
def tan(array):
    """Elementwise tan of an array"""
    return np.tan(array)


@singledispatch
def exp(array):
    """Elementwise exp of an array"""
    return np.exp(array)


@singledispatch
def log(array):
    """Elementwise natural logarithm of an array"""
    return np.log(array)


@singledispatch
def log2(array):
    """Elementwise base-2 logarithm of an array"""
    return np.log2(array)
