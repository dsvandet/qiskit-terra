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

import re

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate, HGate, SGate, TGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix, matrix_equal
from qiskit.quantum_info.operators.matrix import Matrix
from qiskit.quantum_info.dispatch import Array, backend_types


LABEL_MATS = {
    'I': Matrix(IGate(), num_qubits=1, dtype=complex),
    'X': Matrix(XGate(), num_qubits=1, dtype=complex),
    'Y': Matrix(YGate(), num_qubits=1, dtype=complex),
    'Z': Matrix(ZGate(), num_qubits=1, dtype=complex),
    'H': Matrix(HGate(), num_qubits=1, dtype=complex),
    'S': Matrix(SGate(), num_qubits=1, dtype=complex),
    'T': Matrix(TGate(), num_qubits=1, dtype=complex),
    '0': Matrix(np.array([[1, 0], [0, 0]], dtype=complex), num_qubits=1),
    '1': Matrix(np.array([[0, 0], [0, 1]], dtype=complex), num_qubits=1),
    '+': Matrix(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex), num_qubits=1),
    '-': Matrix(np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex), num_qubits=1),
    'r': Matrix(np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex), num_qubits=1),
    'l': Matrix(np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=complex), num_qubits=1),
}
LABEL_REGEX = '^[{}]+$'.format(''.join(LABEL_MATS.keys()))


class Operator(Matrix):
    r"""Matrix operator class

    This represents a matrix operator :math:`M` that will
    :meth:`~Statevector.evolve` a :class:`Statevector` :math:`|\psi\rangle`
    by matrix-vector multiplication

    .. math::

        |\psi\rangle \mapsto M|\psi\rangle,

    and will :meth:`~DensityMatrix.evolve` a :class:`DensityMatrix` :math:`\rho`
    by left and right multiplication

    .. math::

        \rho \mapsto M \rho M^\dagger.
    """

    def __qiskit_operator__(self):
        return self

    def __init__(self, data, input_dims=None, output_dims=None, num_qubits=None):
        """Initialize an operator object.

        Args:
            data (QuantumCircuit or
                  Instruction or
                  BaseOperator or
                  matrix): data to initialize operator.
            input_dims (tuple): the input subsystem dimensions.
                                [Default: None]
            output_dims (tuple): the output subsystem dimensions.
                                 [Default: None]
            num_qubits (int): Optional, number of qubits for the operator.

        Raises:
            QiskitError: if input data cannot be initialized as an operator.

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. If the input data is
            a Numpy array of shape (2**N, 2**N) qubit systems will be used. If
            the input operator is not an N-qubit operator, it will assign a
            single subsystem with dimension specified by the shape of the input.
        """
        if hasattr(data, '__qiskit__operator__'):
            oper = getattr(data, '__qiskit_operator__')()
            if not isinstance(oper, Operator):
                raise QiskitError('object __qiskit_operator__ method not producing an Operator')
            self._array = oper._array
            self._copy_attributes(oper)
            return

        if not isinstance(data, (Matrix, Array, backend_types())):
            if isinstance(data, QuantumCircuit):
                data = self._init_instruction(data)
            elif hasattr(data, 'to_operator'):
                # legacy
                data = data.to_operator()
            elif hasattr(data, 'to_matrix'):
                # legacy
                data = data.to_matrix()
        super().__init__(data, input_dims=input_dims,
                         output_dims=output_dims, num_qubits=num_qubits,
                         dtype=complex)

    @property
    def _data(self):
        # Temporary compatibility until other classes are updated.
        return self.data

    @classmethod
    def from_label(cls, label):
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
        if re.match(LABEL_REGEX, label) is None:
            raise QiskitError('Label contains invalid characters.')
        # Initialize an identity matrix and apply each gate
        num_qubits = len(label)
        op = Operator(np.eye(2 ** num_qubits, dtype=complex))
        for qubit, char in enumerate(reversed(label)):
            if char != 'I':
                op = op.compose(LABEL_MATS[char], qargs=[qubit])
        return op

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return is_unitary_matrix(self._array, rtol=rtol, atol=atol)

    def to_operator(self):
        """Convert operator to matrix operator class"""
        return self

    def to_instruction(self):
        """Convert to a UnitaryGate instruction."""
        # pylint: disable=cyclic-import
        from qiskit.extensions.unitary import UnitaryGate
        return UnitaryGate(np.asarray(self))

    def equiv(self, other, rtol=None, atol=None):
        """Return True if operators are equivalent up to global phase.

        Args:
            other (Operator): an operator object.
            rtol (float): relative tolerance value for comparison.
            atol (float): absolute tolerance value for comparison.

        Returns:
            bool: True if operators are equivalent up to global phase.
        """
        if not isinstance(other, Operator):
            try:
                other = Operator(other)
            except QiskitError:
                return False
        if self.dim != other.dim:
            return False
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return matrix_equal(self.array, other.array, ignore_phase=True,
                            rtol=rtol, atol=atol)

    @classmethod
    def _init_instruction(cls, instruction):
        """Convert a QuantumCircuit or Instruction to an Operator."""
        # Initialize an identity operator of the correct size of the circuit
        dimension = 2 ** instruction.num_qubits
        op = Operator(np.eye(dimension))
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        op._append_instruction(instruction)
        return op

    def _append_instruction(self, obj, qargs=None):
        """Update the current Operator by apply an instruction."""
        from qiskit.circuit.barrier import Barrier
        from .scalar_op import ScalarOp

        if isinstance(obj, Barrier):
            return
        if (hasattr(obj, '__qiskit_operator__') or hasattr(
                obj, '__qiskit_matrix__') or hasattr(obj, '__array__')):
            self._array = self.compose(Operator(obj), qargs=qargs)._array
        else:
            # If the instruction doesn't have a matrix defined we use its
            # circuit decomposition definition if it exists, otherwise we
            # cannot compose this gate and raise an error.
            if obj.definition is None:
                raise QiskitError('Cannot apply Instruction: {}'.format(obj.name))
            if not isinstance(obj.definition, QuantumCircuit):
                raise QiskitError('Instruction "{}" '
                                  'definition is {} but expected QuantumCircuit.'.format(
                                      obj.name, type(obj.definition)))
            if obj.definition.global_phase:
                dimension = 2 ** self.num_qubits
                op = self.compose(
                    ScalarOp(dimension, np.exp(1j * float(obj.definition.global_phase))),
                    qargs=qargs)
                self._array = op._array
            flat_instr = obj.definition.to_instruction()
            for instr, qregs, cregs in flat_instr.definition.data:
                if cregs:
                    raise QiskitError(
                        'Cannot apply instruction with classical registers: {}'.format(
                            instr.name))
                # Get the integer position of the flat register
                if qargs is None:
                    new_qargs = [tup.index for tup in qregs]
                else:
                    new_qargs = [qargs[tup.index] for tup in qregs]
                self._append_instruction(instr, qargs=new_qargs)
