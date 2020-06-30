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
Jax JaxArray single dispatch overloads.
"""

try:
    from jax.interpreters.xla import DeviceArray as JaxArray
    from jax.interpreters.partial_eval import JaxprTracer as JaxParam
    import jax.numpy as jnp
    _HAS_JAX = True

except ImportError:
    _HAS_JAX = False
    JaxArray = None
    JaxParam = None

if _HAS_JAX:

    from .numpy_dispatch import (
        dot, kron, outer, trace, norm, sum, prod, diag, isclose, allclose, real, imag,
        abs, round, transpose, conj, shape, reshape, flatten, ravel, sqrt, sin,
        cos, tan, exp, log, log2
    )

    @dot.register(JaxArray)
    @dot.register(JaxParam)
    def _(array, other):
        return jnp.dot(array, other)

    @kron.register(JaxArray)
    @kron.register(JaxParam)
    def _(array, other):
        return jnp.kron(array, other)

    @outer.register(JaxArray)
    @outer.register(JaxParam)
    def _(array, other):
        return jnp.outer(array, other)

    @trace.register(JaxArray)
    @trace.register(JaxParam)
    def _(array, offset=0, axis1=0, axis2=1, dtype=None):
        return jnp.trace(array, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

    @norm.register(JaxArray)
    @norm.register(JaxParam)
    def _(array, ord=None, axis=None, keepdims=False):
        return jnp.linalg.norm(array, ord=ord, axis=axis, keepdims=keepdims)

    @sum.register(JaxArray)
    @sum.register(JaxParam)
    def _(array, axis=None, dtype=None, keepdims=False):
        return jnp.sum(array, axis=axis, dtype=dtype, keepdims=keepdims)

    @prod.register(JaxArray)
    @prod.register(JaxParam)
    def _(array, axis=None, dtype=None, keepdims=False):
        return jnp.prod(array, axis=axis, dtype=dtype, keepdims=keepdims)

    @diag.register(JaxArray)
    @diag.register(JaxParam)
    def _(arr, k=0):
        return jnp.diag(arr, k=k)

    @isclose.register(JaxArray)
    @isclose.register(JaxParam)
    def _(array, target, rtol=1e-5, atol=1e-8):
        return jnp.isclose(array, target, rtol=rtol, atol=atol)

    @allclose.register(JaxArray)
    @allclose.register(JaxParam)
    def _(array, target, rtol=1e-5, atol=1e-8):
        return jnp.allclose(array, target, rtol=rtol, atol=atol)

    @real.register(JaxArray)
    @real.register(JaxParam)
    def _(array):
        return array.real

    @imag.register(JaxArray)
    @imag.register(JaxParam)
    def _(array):
        return array.imag

    @abs.register(JaxArray)
    @abs.register(JaxParam)
    def _(array):
        return jnp.abs(array)

    @round.register(JaxArray)
    @round.register(JaxParam)
    def _(array, decimals=0):
        return array.round(decimals)

    @transpose.register(JaxArray)
    @transpose.register(JaxParam)
    def _(array, axes=None):
        return array.transpose(axes)

    @conj.register(JaxArray)
    @conj.register(JaxParam)
    def _(array):
        return array.conj()

    @shape.register(JaxArray)
    @shape.register(JaxParam)
    def _(array):
        return array.shape

    @reshape.register(JaxArray)
    @reshape.register(JaxParam)
    def _(array, new_shape, order='C'):
        return array.reshape(new_shape, order=order)

    @flatten.register(JaxArray)
    @flatten.register(JaxParam)
    def _(array, order='C'):
        return array.ravel(order=order)

    @ravel.register(JaxArray)
    @ravel.register(JaxParam)
    def _(array, order='C'):
        return array.ravel(order=order)

    @sqrt.register(JaxArray)
    @sqrt.register(JaxParam)
    def _(array):
        return jnp.sqrt(array)

    @sin.register(JaxArray)
    @sin.register(JaxParam)
    def _(array):
        return jnp.sin(array)

    @cos.register(JaxArray)
    @cos.register(JaxParam)
    def _(array):
        return jnp.cos(array)

    @tan.register(JaxArray)
    @tan.register(JaxParam)
    def _(array):
        return jnp.tan(array)

    @exp.register(JaxArray)
    @exp.register(JaxParam)
    def _(array):
        return jnp.exp(array)

    @log.register(JaxArray)
    @log.register(JaxParam)
    def _(array):
        return jnp.log(array)

    @log2.register(JaxArray)
    @log2.register(JaxParam)
    def _(array):
        return jnp.log2(array)
