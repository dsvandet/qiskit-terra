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
def trace(tensor, offset=0, axis1=0, axis2=1, dtype=None):
    """Trace of a tensor."""
    return np.trace(tensor, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)


@singledispatch
def norm(tensor, ord=None, axis=None, keepdims=False):
    """Return the norm of a tensor."""
    return la.norm(tensor, ord=ord, axis=axis, keepdims=keepdims)


# -------------------------------------------------------------------------
# Array transformation functions
# -------------------------------------------------------------------------

@singledispatch
def dot(tensor, other):
    """Dot product of two tensors."""
    return np.dot(tensor, other)


@singledispatch
def kron(tensor, other):
    """Kronecker product of two tensors."""
    return np.kron(tensor, other)


@singledispatch
def outer(tensor, other):
    """Outer product of two tensors."""
    return np.outer(tensor, other)


@singledispatch
def transpose(tensor, *axes):
    """Transpose a tensor on axes."""
    return tensor.transpose(*axes)


@singledispatch
def conjugate(tensor):
    """Complex conjugate of a tensor."""
    return conj(tensor)


@singledispatch
def conj(tensor):
    """Complex conjugate of a tensor."""
    return tensor.conj()


@singledispatch
def adjoint(tensor, *axes):
    """Complex conjugate transpose of a tensor."""
    return conj(transpose(tensor, *axes))


@singledispatch
def sum(tensor, axis=None, dtype=None, keepdims=False):
    """Sum a tensor along the specified axis."""
    return np.sum(tensor, axis=axis, dtype=dtype, keepdims=keepdims)


@singledispatch
def prod(tensor, axis=None, dtype=None, keepdims=False):
    """Multiply reduce a tensor along the specified axis."""
    return np.prod(tensor, axis=axis, dtype=dtype, keepdims=keepdims)


# -------------------------------------------------------------------------
# Array shape functions
# -------------------------------------------------------------------------

@singledispatch
def shape(tensor):
    """Return the shape of a tensor."""
    return tensor.shape


@singledispatch
def reshape(tensor, new_shape, order='C'):
    """Reshape a tensor."""
    return tensor.reshape(new_shape, order=order)


@singledispatch
def flatten(tensor, order='C'):
    """Flatten a tensor."""
    return tensor.flatten(order=order)


# -------------------------------------------------------------------------
# Array elementwise functions
# -------------------------------------------------------------------------

@singledispatch
def real(tensor):
    """Return the real part of a tensor."""
    return np.real(tensor)


@singledispatch
def imag(tensor):
    """Return the imaginary part of a tensor."""
    return np.imag(tensor)


@singledispatch
def abs(tensor):
    """Return the elementwise absolute value of a tensor."""
    return np.abs(tensor)


@singledispatch
def round(tensor, decimals=0):
    """Round a tensor."""
    return tensor.round(decimals)


@singledispatch
def isclose(tensor, target, rtol=1e-5, atol=1e-8):
    """Sum a tensor along the specified axis."""
    return np.isclose(tensor, target, rtol=rtol, atol=atol)


@singledispatch
def allclose(tensor, target, rtol=1e-5, atol=1e-8):
    """Multiply reduce a tensor along the specified axis."""
    return np.allclose(tensor, target, rtol=rtol, atol=atol)


@singledispatch
def sqrt(tensor):
    """Elementwise sqrt of a tensor."""
    return np.sqrt(tensor)


@singledispatch
def sin(tensor):
    """Elementwise sin of a tensor."""
    return np.sin(tensor)


@singledispatch
def cos(tensor):
    """Elementwise cos of a tensor."""
    return np.cos(tensor)


@singledispatch
def tan(tensor):
    """Elementwise tan of a tensor."""
    return np.tan(tensor)


@singledispatch
def exp(tensor):
    """Elementwise exp of a tensor."""
    return np.exp(tensor)


@singledispatch
def log(tensor):
    """Elementwise natural logarithm of a tensor."""
    return np.log(tensor)


@singledispatch
def log2(tensor):
    """Elementwise base-2 logarithm of a tensor."""
    return np.log2(tensor)
