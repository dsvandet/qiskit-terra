# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Matrix Operator class.
"""

from functools import singledispatch

import qiskit.numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.library.standard_gates import (
    RXGate, RYGate, RZGate, RZXGate, RXXGate, RYYGate, RZZGate)


@singledispatch
def to_matrix(obj, backend='numpy'):
    """Convert gate to an Operator"""
    if not hasattr(obj, 'to_matrix'):
        raise QiskitError('Cannot convert input to a matrix')
    return np.asarray(obj.to_matrix(), backend=backend)


@to_matrix.register(np.ndarray)
def _(obj, backend='numpy'):
    return np.asarray(obj, backend=backend)


@to_matrix.register(np.JaxArray)
@to_matrix.register(np.JaxParam)
def _(obj, backend='numpy'):
    return np.asarray(obj, backend=backend)


@to_matrix.register(BaseOperator)
def _(obj, backend='numpy'):
    return np.asarray(obj.to_operator().data, backend=backend)


@to_matrix.register(RXGate)
def _(obj, backend='numpy'):
    print('TO_MATRIX RX')
    theta2 = obj.params[0] / 2
    mat = np.cos(theta2) * np.asarray(
        [[1., 0.], [0., 1.]], backend=backend) + -1j * np.sin(theta2) * np.asarray(
            [[0., 1.], [1., 0.]], backend=backend
        )
    print('TO_MATRIX RX OK')
    return mat


@to_matrix.register(RYGate)
def _(obj, backend='numpy'):
    print('TO_MATRIX RY')
    theta2 = obj.params[0] / 2
    mat = np.cos(theta2) * np.asarray(
        [[1., 0.], [0., 1.]], backend=backend) + np.sin(theta2) * np.asarray(
            [[0., -1.], [1., 0.]], backend=backend
        )
    print('TO_MATRIX RY OK')
    return mat


@to_matrix.register(RZGate)
def _(obj, backend='numpy'):
    theta2 = obj.params[0] / 2
    diag = np.exp(1j * theta2 * np.asarray(
        [-1., 1.], backend=backend))
    return np.diag(diag)


@to_matrix.register(RXXGate)
def _(obj, backend='numpy'):
    theta2 = obj.params[0] / 2
    return np.cos(theta2) * np.asarray(
        np.eye(4), backend=backend) - 1j * np.sin(theta2) * np.asarray(
            [0., 0., 0., 1.], [0., 0., 1., 0.],
            [0., 1., 0., 0.], [1., 0., 0., 0.],
            backend=backend
        )


@to_matrix.register(RYYGate)
def _(obj, backend='numpy'):
    theta2 = obj.params[0] / 2
    return np.cos(theta2) * np.asarray(
        np.eye(4), backend=backend) + 1j * np.sin(theta2) * np.asarray(
            [0., 0., 0., 1.], [0., 0., -1., 0.],
            [0., -1., 0., 0.], [1., 0., 0., 0.],
            backend=backend
        )


@to_matrix.register(RZZGate)
def _(obj, backend='numpy'):
    theta2 = obj.params[0] / 2
    diag = np.exp(1j * theta2 * np.asarray(
        [-1., 1., 1., -1.], backend=backend))
    return np.diag(diag)


@to_matrix.register(RZXGate)
def _(obj, backend='numpy'):
    theta2 = obj.params[0] / 2
    return np.cos(theta2) * np.asarray(
        np.eye(4), backend=backend) + 1j * np.sin(theta2) * np.asarray(
            [0., 0., -1., 0.], [0., 0., 0., 1.],
            [-1., 0., 0., 0.], [0., 1., 0., 0.],
            backend=backend
        )
