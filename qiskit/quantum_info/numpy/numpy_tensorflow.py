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
Tensorflow Tensor single dispatch overloads.
"""

from qiskit.exceptions import QiskitError

try:
    import tensorflow as tf
    from tensorflow.python.framework.ops import Tensor as TensorflowArray
    _HAS_TENSORFLOW = True

except ImportError:
    _HAS_TENSORFLOW = False
    TensorflowArray = None

if _HAS_TENSORFLOW:

    from .numpy_dispatch import (
        dot, kron, outer, trace, norm, sum, prod, real, imag,
        abs, round, transpose, conj, adjoint, shape, reshape, flatten, sqrt, sin,
        cos, tan, exp, log, log2
    )

    @dot.register(TensorflowArray)
    def _(tensor, other):
        return tf.tensordot(tensor, other, axes=1)

    @outer.register(TensorflowArray)
    def _(tensor, other):
        return tf.tensordot(tensor, other, axes=0)

    @kron.register(TensorflowArray)
    def _(tensor, other):
        val = tf.tensordot(tensor, other, axes=0)
        if tensor.ndim == 1:
            new_shape = list(other.shape)
            new_shape[0] = new_shape[0] * tensor.shape[0]
            return tf.reshape(val, new_shape)

        if tensor.ndim == 2:
            if other.ndim == 2:
                perm = [0, 2, 1, 3]
                new_shape = [tensor.shape[0] + other.shape[0],
                             tensor.shape[1] + other.shape[1]]
                return tf.reshape(tf.transpose(val, perm=perm), new_shape)
            if other.ndim == 1:
                perm = [0, 2, 1]
                new_shape = [tensor.shape[0] + other.shape[0], tensor.shape[1]]
                return tf.reshape(tf.transpose(val, perm=perm), new_shape)
        raise QiskitError("Tensorflow kron overload is only defined for matrix and vector tensors.")

    @trace.register(TensorflowArray)
    def _(tensor, offset=0, axis1=0, axis2=1, dtype=None):
        return tf.linalg.trace(tensor, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

    @norm.register(TensorflowArray)
    def _(tensor, ord=None, axis=None, keepdims=False):
        return tf.norm(tensor, ord=ord, axis=axis, keepdims=keepdims)

    @sum.register(TensorflowArray)
    def _(tensor, axis=None, dtype=None, keepdims=False):
        # Ignores dtype argument
        return tf.math.reduce_sum(tensor, axis=axis, keepdims=keepdims)

    @prod.register(TensorflowArray)
    def _(tensor, axis=None, dtype=None, keepdims=False):
        return tf.math.reduce_sum(tensor, axis=axis, keepdims=keepdims)

    @real.register(TensorflowArray)
    def _(tensor):
        return tf.math.real(tensor)

    @imag.register(TensorflowArray)
    def _(tensor):
        return tf.math.imag(tensor)

    @abs.register(TensorflowArray)
    def _(tensor):
        return tf.abs(tensor)

    @round.register(TensorflowArray)
    def _(tensor, decimals=0):
        scale = tf.constant(10 ** decimals, dtype=tensor.dtype)
        return tf.round(tensor * scale) / scale

    @transpose.register(TensorflowArray)
    def _(tensor, *axes):
        if axes:
            perm = axes[0]
        else:
            perm = None
        return tf.transpose(tensor, perm=perm)

    @adjoint.register(TensorflowArray)
    def _(tensor, *axes):
        if axes:
            perm = axes[0]
        else:
            perm = None
        return tf.transpose(tensor, perm=perm, conjugate=True)

    @conj.register(TensorflowArray)
    def _(tensor):
        return tf.math.conj(tensor)

    @shape.register(TensorflowArray)
    def _(tensor):
        return tensor.shape

    @reshape.register(TensorflowArray)
    def _(tensor, new_shape, order='C'):
        # Tensorflow doesn't support F ordering
        if order == 'F':
            if tensor.ndim == 1:
                return tf.transpose(tf.reshape(tensor, new_shape))
            return tf.reshape(transpose(tensor), new_shape)
        return tf.reshape(tensor, new_shape)

    @flatten.register(TensorflowArray)
    def _(tensor, order='C'):
        if order == 'F':
            tensor = tf.transpose(tensor)
        return tf.reshape(tensor, tf.size(tensor).numpy())

    @sqrt.register(TensorflowArray)
    def _(tensor):
        return tf.math.sqrt(tensor)

    @sin.register(TensorflowArray)
    def _(tensor):
        return tf.math.sin(tensor)

    @cos.register(TensorflowArray)
    def _(tensor):
        return tf.math.cos(tensor)

    @tan.register(TensorflowArray)
    def _(tensor):
        return tf.math.tan(tensor)

    @exp.register(TensorflowArray)
    def _(tensor):
        return tf.math.exp(tensor)

    @log.register(TensorflowArray)
    def _(tensor):
        return tf.math.log(tensor)

    @log2.register(TensorflowArray)
    def _(tensor):
        return log(tensor) / log(2)
