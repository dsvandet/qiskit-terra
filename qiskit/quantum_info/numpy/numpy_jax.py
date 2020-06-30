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
        dot, kron, outer, trace, norm, sum, prod, isclose, allclose, real, imag,
        abs, round, transpose, conj, shape, reshape, flatten, sqrt, sin,
        cos, tan, exp, log, log2
    )

    @dot.register(JaxArray)
    @dot.register(JaxParam)
    def _(tensor, other):
        return jnp.dot(tensor, other)

    @kron.register(JaxArray)
    @kron.register(JaxParam)
    def _(tensor, other):
        return jnp.kron(tensor, other)

    @outer.register(JaxArray)
    @outer.register(JaxParam)
    def _(tensor, other):
        return jnp.outer(tensor, other)

    @trace.register(JaxArray)
    @trace.register(JaxParam)
    def _(tensor, offset=0, axis1=0, axis2=1, dtype=None):
        return jnp.trace(tensor, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

    @norm.register(JaxArray)
    @norm.register(JaxParam)
    def _(tensor, ord=None, axis=None, keepdims=False):
        return jnp.linalg.norm(tensor, ord=ord, axis=axis, keepdims=keepdims)

    @sum.register(JaxArray)
    @sum.register(JaxParam)
    def _(tensor, axis=None, dtype=None, keepdims=False):
        return jnp.sum(tensor, axis=axis, dtype=dtype, keepdims=keepdims)

    @prod.register(JaxArray)
    @prod.register(JaxParam)
    def _(tensor, axis=None, dtype=None, keepdims=False):
        return jnp.prod(tensor, axis=axis, dtype=dtype, keepdims=keepdims)

    @isclose.register(JaxArray)
    @isclose.register(JaxParam)
    def _(tensor, target, rtol=1e-5, atol=1e-8):
        return jnp.isclose(tensor, target, rtol=rtol, atol=atol)

    @allclose.register(JaxArray)
    @allclose.register(JaxParam)
    def _(tensor, target, rtol=1e-5, atol=1e-8):
        return jnp.allclose(tensor, target, rtol=rtol, atol=atol)

    @real.register(JaxArray)
    @real.register(JaxParam)
    def _(tensor):
        return tensor.real

    @imag.register(JaxArray)
    @imag.register(JaxParam)
    def _(tensor):
        return tensor.imag

    @abs.register(JaxArray)
    @abs.register(JaxParam)
    def _(tensor):
        return jnp.abs(tensor)

    @round.register(JaxArray)
    @round.register(JaxParam)
    def _(tensor, decimals=0):
        return tensor.round(decimals)

    @transpose.register(JaxArray)
    @transpose.register(JaxParam)
    def _(tensor, *axes):
        return tensor.transpose(*axes)

    @conj.register(JaxArray)
    @conj.register(JaxParam)
    def _(tensor):
        return tensor.conj()

    @shape.register(JaxArray)
    @shape.register(JaxParam)
    def _(tensor):
        return tensor.shape

    @reshape.register(JaxArray)
    @reshape.register(JaxParam)
    def _(tensor, new_shape, order='C'):
        return tensor.reshape(new_shape, order=order)

    @flatten.register(JaxArray)
    @flatten.register(JaxParam)
    def _(tensor, order='C'):
        return tensor.ravel(order=order)

    @sqrt.register(JaxArray)
    @sqrt.register(JaxParam)
    def _(tensor):
        return jnp.sqrt(tensor)

    @sin.register(JaxArray)
    @sin.register(JaxParam)
    def _(tensor):
        return jnp.sin(tensor)

    @cos.register(JaxArray)
    @cos.register(JaxParam)
    def _(tensor):
        return jnp.cos(tensor)

    @tan.register(JaxArray)
    @tan.register(JaxParam)
    def _(tensor):
        return jnp.tan(tensor)

    @exp.register(JaxArray)
    @exp.register(JaxParam)
    def _(tensor):
        return jnp.exp(tensor)

    @log.register(JaxArray)
    @log.register(JaxParam)
    def _(tensor):
        return jnp.log(tensor)

    @log2.register(JaxArray)
    @log2.register(JaxParam)
    def _(tensor):
        return jnp.log2(tensor)
