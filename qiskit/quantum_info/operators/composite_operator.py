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

import copy
from numbers import Number

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp


class CompositeOperator(BaseOperator):
    """Composite operator class"""

    def __init__(self, ops, coeffs=None, dims=None):
        """Initialize an operator object."""

        # Initialize operator list
        if isinstance(ops, (np.ndarray, QuantumCircuit, Instruction)):
            self._ops = [Operator(ops)]
        elif isinstance(ops, CompositeOperator):
            self._ops = ops._ops
            if coeffs is None:
                coeffs = ops._coeffs
            if dims is None:
                dims = ops._input_dims
        elif isinstance(ops, BaseOperator):
            self._ops = [ops]
        else:
            # TODO: conversion of other types
            self._ops = ops
        # Check qargs are defined for all operators and if not add them
        for i, op in enumerate(self._ops):
            if op.qargs is None:
                qargs = list(range(len(ops._input_dims)))
                self._ops[i] = op(qargs)

        # Add coefficients
        if coeffs is None:
            self._coeffs = len(self._ops) * [1]
        else:
            if isinstance(coeffs, Number):
                coeffs = [coeffs]
            self._coeffs = list(coeffs)
            if len(self._coeffs) != len(self._ops):
                raise QiskitError("Length of coefficients don't match number of ops")

        # Compute dimensions for ops
        if dims is None:
            dims_dict = {}
            for op in self._ops:
                dims = op._input_dims
                if op._output_dims != dims:
                    raise QiskitError("Data contains a non-square operator")
                for dim, qarg in zip(dims, op.qargs):
                    if qarg in dims_dict and dim != dims_dict[qarg]:
                        raise QiskitError("Dimensions differ accross components")
                    else:
                        dims_dict[qarg] = dim
            # Convert dimensions into a list
            nqargs = max(dims_dict) + 1
            if len(dims_dict) != nqargs:
                raise QiskitError("Cannot automatically compute dims")
            dims = nqargs * [0]
            for qarg, dim in dims_dict.items():
                dims[qarg] = dim

        # TODO: Validate dims

        # Initialize BaseOperator dimensions
        dims = self._automatic_dims(dims, np.product(dims))
        super().__init__(dims, dims)

    def __repr__(self):
        prefix = 'CompositeOperator('
        ops = [(op.qargs, op.__class__.__name__) for op in self._ops]
        if len(ops) == 0:
            return '{}[], dims={})'.format(prefix, self._input_dims)
        pad = len(prefix) * ' '
        return '{}{},\n{}coeffs={}, dims={})'.format(
            prefix, ops, pad, self._coeffs, self._input_dims)

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        # Check equal by converting to Operators
        return self.to_operator() == other.to_operator()

    @property
    def ops(self):
        """Return ops."""
        return self._ops

    @property
    def coeffs(self):
        """Return coeffs."""
        return self._coeffs

    @staticmethod
    def from_label(label):
        """Return a tensor product of single-qubit operators.

        Args:
            label (string): single-qubit operator string.

        Returns:
            Operator: The N-qubit operator.

        Raises:
            QiskitError: if the label contains invalid characters, or the
                         length of the label is larger than an explicitly
                         specified num_qubits.

        Additional Information:
            The labels correspond to the single-qubit matrices:
            'I': [[1, 0], [0, 1]]
            'X': [[0, 1], [1, 0]]
            'Y': [[0, -1j], [1j, 0]]
            'Z': [[1, 0], [0, -1]]
            'H': [[1, 1], [1, -1]] / sqrt(2)
            'S': [[1, 0], [0 , 1j]]
            'T': [[1, 0], [0, (1+1j) / sqrt(2)]]
            '0': [[1, 0], [0, 0]]
            '1': [[0, 0], [0, 1]]
            '+': [[0.5, 0.5], [0.5 , 0.5]]
            '-': [[0.5, -0.5], [-0.5 , 0.5]]
            'r': [[0.5, -0.5j], [0.5j , 0.5]]
            'l': [[0.5, 0.5j], [-0.5j , 0.5]]
        """
        return CompositeOperator(Operator.from_label(label))

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        return self.to_operator().is_unitary(atol=atol, rtol=rtol)

    def to_operator(self):
        """Convert operator to matrix operator class"""
        ret = Operator(
            np.zeros(2 * [self._input_dim], dtype=complex),
            input_dims=self._input_dims, output_dims=self._output_dims)
        for coeff, op in zip(self._coeffs, self._ops):
            ret = ret + (coeff * op)(op.qargs)
        return ret

    def to_matrix(self):
        """Convert to a Numpy array"""
        return self.to_operator().data

    def to_instruction(self):
        """Convert to a UnitaryGate instruction."""
        return self.to_operator().to_instruction()

    def conjugate(self):
        """Return the conjugate of the operator."""
        # Make a shallow copy and update array
        # TODO: Check qargs is preserved by conjugate for
        # all used operator subclasses (Ok for Operator)
        ret = copy.copy(self)
        ret._coeffs = self._conjugate_coeffs(self.coeffs)
        ret._ops = [op.conjugate() for op in self.ops]
        return ret

    def transpose(self):
        """Return the transpose of the operator."""
        ret = copy.copy(self)
        ret._ops = [op.transpose() for op in self.ops]
        return ret

    def adjoint(self):
        """Return the adjoint of the operator."""
        ret = copy.copy(self)
        ret._coeffs = self._conjugate_coeffs(self.coeffs)
        ret._ops = [op.adjoint() for op in self.ops]
        return ret

    def compose(self, other, qargs=None, front=False):
        """Return the composed operator.

        Args:
            other (CompositeOperator): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            CompositeOperator: The operator self @ other.

        Raise:
            QiskitError: if operators have incompatible dimensions for
                         composition.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            matrix operators. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, CompositeOperator):
            other = CompositeOperator(other)

        # Validate dimensions are compatible and return the composed
        # operator dimensions
        input_dims, _ = self._get_compose_dims(
            other, qargs, front)

        coeffs = []
        ops = []
        for coeff_a, op_a in zip(self._coeffs, self._ops):
            for coeff_b, op_b in zip(other._coeffs, other._ops):
                coeff = coeff_a * coeff_b
                # Only add term if the coefficient is non-zero
                if not CompositeOperator._is_zero_coeff(coeff):
                    # combine qargs
                    qargs = list(set(op_a.qargs + op_b.qargs))
                    # combine dims
                    init = ScalarOp(self.input_dims(qargs))
                    ops.append(init.compose(op_a).compose(op_b, front=front)(qargs))
                    coeffs.append(coeff)

        return CompositeOperator(ops, coeffs, dims=self._input_dims)

    def dot(self, other, qargs=None):
        """Return the right multiplied operator self * other.

        Args:
            other (ComppositeOperator): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].

        Returns:
            CompositeOperator: The operator self * other.

        Raises:
            QiskitError: if other cannot be converted to an Operator or has
                         incompatible dimensions.
        """
        return super().dot(other, qargs=qargs)

    def tensor(self, other):
        """Return the tensor product operator self ⊗ other.

        Args:
            other (CompositeOperator): a operator subclass object.

        Returns:
            CompositeOperator: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other cannot be converted to an operator.
        """
        if not isinstance(other, CompositeOperator):
            other = CompositeOperator(other)
        return self._tensor_product(self, other)

    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Args:
            other (CompositeOperator): an operator object.

        Returns:
            CompositeOperator: the tensor product operator other ⊗ self.

        Raises:
            QiskitError: if other cannot be converted to an operator.
        """
        if not isinstance(other, CompositeOperator):
            other = CompositeOperator(other)
        return self._tensor_product(other, self)

    def _add(self, other, qargs=None):
        """Return the operator self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (CompositeOperator): an operator object.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            CompositeOperator: the operator self + other.

        Raises:
            QiskitError: if other is not an operator, or has incompatible
                         dimensions.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, CompositeOperator):
            other = CompositeOperator(other)

        self._validate_add_dims(other, qargs)

        ret = copy.copy(self)
        ret._coeffs = self._coeffs + other._coeffs
        if qargs is None:
            ret._ops = self._ops + other._ops
        else:
            # If adding on qargs we remap qubits in other
            new_ops = copy.copy(other._ops)
            for i, op in enumerate(other._ops):
                new_qargs = [qargs[i] for i in op.qargs]
                new_ops[i] = op(new_qargs)
            ret._ops = self._ops + new_ops
        return ret

    def _multiply(self, other):
        """Return the operator self * other.

        Args:
            other (complex): a complex number.

        Returns:
            CompositeOperator: the operator other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")

        # Check if multiplying by zero
        if self._is_zero_coeff(other):
            return CompositeOperator([], dims=self._input_dims)

        ret = copy.copy(self)
        ret._coeffs = [other * coeff for coeff in self._coeffs]
        return ret

    def equiv(self, other, rtol=None, atol=None):
        """Return True if operators are equivalent up to global phase.

        Args:
            other (CompositeOperator): an operator object.
            rtol (float): relative tolerance value for comparison.
            atol (float): absolute tolerance value for comparison.

        Returns:
            bool: True if operators are equivalent up to global phase.
        """
        return self.to_operator().equiv(other, rtol=rtol, atol=atol)

    @staticmethod
    def _conjugate_coeffs(coeffs):
        """Conjugate a Coefficient object"""
        # TODO: Allow for other coeff types here
        return np.conj(coeffs).tolist()

    @classmethod
    def _is_zero_coeff(cls, coeff, atol=None, rtol=None):
        """Check if a coefficient object is zero"""
        if atol is None:
            atol = cls._ATOL_DEFAULT
        if rtol is None:
            rtol = cls._ATOL_DEFAULT

        if isinstance(coeff, Number):
            return np.isclose(coeff, 0, atol=atol, rtol=rtol)

        # TODO: Allow for other coeff types here
        raise QiskitError("Invalid coefficient")

    @staticmethod
    def _tensor_product(a, b):
        """Tensor product two CompositeOperators"""
        dims = b._input_dims + a._input_dims
        shift = len(b._input_dims)
        coeffs = []
        ops = []
        for coeff_a, op_a in zip(a._coeffs, a._ops):
            for coeff_b, op_b in zip(b._coeffs, b._ops):
                coeff = coeff_a * coeff_b
                # Only add term if the coefficient is non-zero
                if not CompositeOperator._is_zero_coeff(coeff):
                    qargs = op_b.qargs + [i + shift for i in op_a.qargs]
                    ops.append(op_a.tensor(op_b)(qargs))
                    coeffs.append(coeff)
        return CompositeOperator(ops, coeffs, dims=dims)

    # These inplace functions are missing from BaseOperator
    def __iadd__(self, other):
        tmp = self._add(other)
        self._coeffs = tmp._coeffs
        self._ops = tmp._ops
        return self

    def __isub__(self, other):
        tmp = self - other
        self._coeffs = tmp._coeffs
        self._ops = tmp._ops
        return self

    def __imul__(self, other):
        tmp = other * self
        self._coeffs = tmp._coeffs
        self._ops = tmp._ops
        return self
