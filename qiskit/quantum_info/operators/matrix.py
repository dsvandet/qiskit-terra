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
Matrix Matrix class.
"""

import copy
from typing import Optional, Union, Tuple
from numbers import Number

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.dispatch import Array, asarray, backend_types


def implements(np_function, container):
    """Register an __array_function__ implementation."""
    def decorator(func):
        container[np_function] = func
        return func
    return decorator


class Matrix(BaseOperator):
    r"""Matrix operator class

    This represents multi-partite matrix :math:`M`.
    """
    # Numpy Compatibility
    _HANDLED_TYPES = backend_types() + (Array, )
    _HANDLED_FUNCTIONS = {}
    _DELEGATE_FUNCTIONS = {
        np.trace,
        np.linalg.cholesky,
        np.linalg.qr,
        np.linalg.svd,
        np.linalg.eig,
        np.linalg.eigh,
        np.linalg.eigvals,
        np.linalg.eigvalsh,
        np.linalg.norm,
        np.linalg.cond,
        np.linalg.det,
        np.linalg.matrix_rank,
        np.linalg.slogdet,
        np.linalg.solve,
        np.linalg.lstsq,
        np.linalg.inv,
        np.linalg.pinv
    }

    def __array__(self):
        return self._array.__array__()

    def __qiskit_array__(self):
        return self._array

    def __qiskit_matrix__(self):
        if self.__class__ is Matrix:
            return self
        # This allows the function to also work for Matrix subclasses
        return Matrix(self._array, input_dims=self._input_dims,
                      output_dims=self._output_dims,
                      num_qubits=self.num_qubits)

    def __init__(self,
                 data: Array,
                 input_dims: Optional[Union[int, Tuple[int]]] = None,
                 output_dims: Optional[Union[int, Tuple[int]]] = None,
                 num_qubits: Optional[int] = None,
                 dtype: Optional[any] = None,
                 backend: Optional[str] = None):
        """Initialize an operator object.

        Args:
            data (matrix_like): matrix to initialize operator.
            output_dims : Optional, the output subsystem dimensions.
            input_dims: Optional, the input subsystem dimensions.
            num_qubits: Optional, number of qubits for the operator.
            dtype: Optional. The dtype of the returned array. This value
                   must be supported by the specified array backend.
            backend: A registered array backend name. If None the
                     default array backend will be used.

        Raises:
            QiskitError: if input data cannot be initialized as an operator.


        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. If the input data is
            a Numpy array of shape (2**N, 2**N) qubit systems will be used. If
            the input operator is not an N-qubit operator, it will assign a
            single subsystem with dimension specified by the shape of the input.
        """
        if hasattr(data, '__qiskit_matrix__'):
            matrix = data.__qiskit_matrix__()
            if not isinstance(matrix, Matrix):
                raise QiskitError('object __qiskit_matrix__ method not producing a Matrix')
            self._array = matrix._array
            self._copy_attributes(matrix)
            # Convert dtype and backend if specified
            if backend or dtype:
                self._array = asarray(self._array, dtype=dtype, backend=backend)
            return

        self._array = Array(data, dtype=dtype, backend=backend)
        super().__init__(**self._validate_dims(np.shape(self._array),
                                               input_dims=input_dims,
                                               output_dims=output_dims,
                                               num_qubits=num_qubits))

    def __repr__(self):
        prefix = type(self).__name__ + '('
        if self.num_qubits:
            suffix = ', num_qubits={})'.format(self.num_qubits)
        else:
            suffix = ', output_dims={}, input_dims={})'.format(
                self.input_dims(), self.output_dims())
        max_line_width = (np.get_printoptions()['linewidth'] + len(suffix)
                          + len(prefix))
        array_str = np.array2string(
            self.data, separator=', ', prefix=prefix, suffix=suffix,
            max_line_width=max_line_width)
        return prefix + array_str + suffix

    def __eq__(self, other):
        """Test if two Matrices are equal."""
        if not super().__eq__(other):
            return False
        return np.allclose(
            self._array, other._array, rtol=self.rtol, atol=self.atol)

    def __getitem__(self, key):
        return self._array[key]

    def __setitem__(self, key, value):
        self._array[key] = value

    @property
    def array(self):
        """Return the interal Array."""
        return self._array

    @array.setter
    def array(self, value):
        """Update the internal Array."""
        self._array[:] = value

    @property
    def data(self):
        """Return the interal Array data."""
        return self._array.data

    @data.setter
    def data(self, value):
        """Update the internal Array data."""
        self._array.data = value

    def copy(self):
        """Make a deepcopy of the array"""
        ret = copy.copy(self)
        ret._array = np.copy(self._array)
        return ret

    @property
    def shape(self):
        """Return the matrix shape."""
        return np.shape(self._array)

    @property
    def tensor_shape(self):
        """Return the tensor shape of the multi-partite matrix."""
        return tuple(reversed(self.output_dims())) + tuple(
            reversed(self.input_dims()))

    @implements(np.conjugate, _HANDLED_FUNCTIONS)
    def conjugate(self):
        """Return the conjugate of the operator."""
        # Make a shallow copy and update array
        ret = copy.copy(self)
        ret._array = np.conj(self._array)
        return ret

    @implements(np.transpose, _HANDLED_FUNCTIONS)
    def transpose(self):
        """Return the transpose of the operator."""
        # Make a shallow copy and update array
        ret = copy.copy(self)
        ret._array = np.transpose(self._array)
        # Swap input and output dimensions
        if not self.num_qubits:
            ret._set_dims(self.output_dims(), self.input_dims())
        return ret

    def compose(self, other, qargs=None, front=False):
        """Return the composed operator.

        Args:
            other (Matrix): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            Matrix: The operator self @ other.

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
        if not isinstance(other, Matrix):
            other = Matrix(other)
        # Validate dimensions are compatible and return the composed
        # operator dimensions
        kwargs = self._get_compose_dims(other, qargs, front)

        # Full composition of operators
        if qargs is None:
            if front:
                # Composition self * other
                data = np.dot(self._array, other._array)
            else:
                # Composition other * self
                data = np.dot(other._array, self._array)
            return type(self)(data, **kwargs)

        # Compose with other on subsystem
        if front:
            num_indices = self._num_input
            shift = self._num_output
            right_mul = True
        else:
            num_indices = self._num_output
            shift = 0
            right_mul = False

        # Reshape current matrix
        # Note that we must reverse the subsystem dimension order as
        # qubit 0 corresponds to the right-most position in the tensor
        # product, which is the last tensor wire index.
        tensor = np.reshape(self._array, self.tensor_shape)
        mat = np.reshape(other._array, other.tensor_shape)
        indices = [num_indices - 1 - qubit for qubit in qargs]
        if kwargs['num_qubits']:
            num_qubits = kwargs['num_qubits']
            final_shape = (2 ** num_qubits, ) * 2
        else:
            final_shape = [np.product(kwargs['output_dims']),
                           np.product(kwargs['input_dims'])]
        data = np.reshape(
            self._einsum_matmul(tensor, mat, indices, shift, right_mul),
            final_shape)
        return type(self)(data, **kwargs)

    @implements(np.dot, _HANDLED_FUNCTIONS)
    def dot(self, other, qargs=None):
        """Return the right multiplied operator self * other.

        Args:
            other (Matrix): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].

        Returns:
            Matrix: The operator self * other.

        Raises:
            QiskitError: if other cannot be converted to an Matrix or has
                         incompatible dimensions.
        """
        return super().dot(other, qargs=qargs)

    @implements(np.linalg.matrix_power, _HANDLED_FUNCTIONS)
    def power(self, n):
        """Return the matrix power of the operator.

        Args:
            n (int): the power to raise the matrix to.

        Returns:
            BaseOperator: the n-times composed operator.

        Raises:
            QiskitError: if the input and output dimensions of the operator
                         are not equal, or the power is not a positive integer.
        """
        if self.num_qubits is None and self.input_dims() != self.output_dims():
            raise QiskitError("Can only take power of square Matrix.")
        ret = copy.copy(self)
        ret._array = np.linalg.matrix_power(self._array, n)
        return ret

    @implements(np.kron, _HANDLED_FUNCTIONS)
    def tensor(self, other):
        """Return the tensor product operator self ⊗ other.

        Args:
            other (Matrix): a operator subclass object.

        Returns:
            Matrix: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other cannot be converted to an operator.
        """
        if not isinstance(other, Matrix):
            other = Matrix(other)
        data = np.kron(self._array, other._array)
        if self.num_qubits and other.num_qubits:
            return type(self)(data, num_qubits=self.num_qubits + other.num_qubits)
        else:
            input_dims = other.input_dims() + self.input_dims()
            output_dims = other.output_dims() + self.output_dims()
            return type(self)(data, input_dims=input_dims, output_dims=output_dims)

    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Args:
            other (Matrix): an operator object.

        Returns:
            Matrix: the tensor product operator other ⊗ self.

        Raises:
            QiskitError: if other cannot be converted to an operator.
        """
        if not isinstance(other, Matrix):
            other = Matrix(other)
        data = np.kron(other._array, self._array)
        if self.num_qubits and other.num_qubits:
            return type(self)(data, num_qubits=self.num_qubits + other.num_qubits)
        else:
            input_dims = self.input_dims() + other.input_dims()
            output_dims = self.output_dims() + other.output_dims()
            return type(self)(data, input_dims=input_dims, output_dims=output_dims)

    def _add(self, other, qargs=None):
        """Return the operator self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (Matrix): an operator object.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            Matrix: the operator self + other.

        Raises:
            QiskitError: if other is not an operator, or has incompatible
                         dimensions.
        """
        # pylint: disable=import-outside-toplevel, cyclic-import
        from qiskit.quantum_info.operators.scalar_op import ScalarOp

        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, Matrix):
            other = Matrix(other)

        self._validate_add_dims(other, qargs)
        other = ScalarOp._pad_with_identity(self, other, qargs)

        ret = copy.copy(self)
        ret._array = self.array + other.array
        return ret

    def _multiply(self, other):
        """Return the operator self * other.

        Args:
            other (complex): a complex number.

        Returns:
            Operator: the operator other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        ret = copy.copy(self)
        ret._array = other * self._array
        return ret

    @implements(np.around, _HANDLED_FUNCTIONS)
    @implements(np.round_, _HANDLED_FUNCTIONS)
    def round(self, decimals=0):
        """Round an Matrix to the given number of decimals."""
        ret = copy.copy(self)
        ret._array = np.around(ret._array, decimals=decimals)
        return ret

    @implements(np.linalg.inv, _HANDLED_FUNCTIONS)
    def inverse(self):
        """Compute the inverse of the matrix."""
        ret = copy.copy(self)
        ret._array = np.linalg.inv(self._array)
        # Swap input and output dimensions
        if not self.num_qubits:
            ret._set_dims(self.output_dims(), self.input_dims())
        return ret

    @classmethod
    def _einsum_matmul(cls, tensor, mat, indices, shift=0, right_mul=False):
        """Perform a contraction using Numpy.einsum

        Args:
            tensor (np.array): a vector or matrix reshaped to a rank-N tensor.
            mat (np.array): a matrix reshaped to a rank-2M tensor.
            indices (list): tensor indices to contract with mat.
            shift (int): shift for indices of tensor to contract [Default: 0].
            right_mul (bool): if True right multiply tensor by mat
                              (else left multiply) [Default: False].

        Returns:
            Numpy.ndarray: the matrix multiplied rank-N tensor.

        Raises:
            QiskitError: if mat is not an even rank tensor.
        """
        rank = tensor.ndim
        rank_mat = mat.ndim
        if rank_mat % 2 != 0:
            raise QiskitError(
                "Contracted matrix must have an even number of indices.")
        # Get einsum indices for tensor
        indices_tensor = list(range(rank))
        for j, index in enumerate(indices):
            indices_tensor[index + shift] = rank + j
        # Get einsum indices for mat
        mat_contract = list(reversed(range(rank, rank + len(indices))))
        mat_free = [index + shift for index in reversed(indices)]
        if right_mul:
            indices_mat = mat_contract + mat_free
        else:
            indices_mat = mat_free + mat_contract
        return np.einsum(tensor, indices_tensor, mat, indices_mat)

    @classmethod
    def _validate_dims(cls, shape, input_dims=None,
                       output_dims=None, num_qubits=None):
        """Check dimensions and return kwargs for BaseOperator init."""
        if len(shape) != 2:
            raise QiskitError(
                "Input Array is not a Matrix (shape = {})".format(shape))

        # Check qubit case
        if num_qubits:
            if (shape[1] != shape[0] or shape[0] != 2 ** num_qubits):
                raise QiskitError("num_qubits does not match matrix size.")
            return {'input_dims': None,
                    'output_dims': None,
                    'num_qubits': num_qubits}

        # Check general case
        input_size = shape[1]
        output_size = shape[0]
        if input_dims is None:
            input_dims = input_size
            din_int = True
        else:
            din_int = isinstance(input_dims, (int, np.integer))
            if np.product(input_dims) != input_size:
                raise QiskitError("Input dimensions do not match size.")
        if output_dims is None:
            output_dims = output_size
            dout_int = True
        else:
            dout_int = isinstance(output_dims, (int, np.integer))
            if np.product(output_dims) != output_size:
                raise QiskitError("Output dimensions do not match size.")

        # Check if N-qubit input
        if din_int:
            num_qubits = int(np.log2(input_dims))
            if input_size == 2 ** num_qubits:
                # Check if square
                if dout_int and output_size == input_size:
                    return {'input_dims': None,
                            'output_dims': None,
                            'num_qubits': num_qubits}
                else:
                    input_dims = (2, ) * num_qubits
            else:
                input_dims = (input_dims, )
        else:
            input_dims = tuple(input_dims)

        # Check if N-qubit output
        if dout_int:
            num_qubits = int(np.log2(output_dims))
            if output_size == 2 ** num_qubits:
                output_dims = (2, ) * num_qubits
            else:
                output_dims = (output_dims, )
        else:
            output_dims = tuple(output_dims)

        return {'input_dims': input_dims,
                'output_dims': output_dims,
                'num_qubits': None}

    def _get_compose_dims(self, other, qargs, front):
        """Check subsystems are compatible for composition."""
        # Check if both qubit operators
        if self.num_qubits and other.num_qubits:
            if qargs and other.num_qubits != len(qargs):
                raise QiskitError(
                    "Other operator number of qubits does not match the "
                    "number of qargs ({} != {})".format(
                        other.num_qubits, len(qargs)))
            if qargs is None and self.num_qubits != other.num_qubits:
                raise QiskitError(
                    "Other operator number of qubits does not match the "
                    "current operator ({} != {})".format(
                        other.num_qubits, self.num_qubits))
            return {'input_dims': None, 'output_dims': None,
                    'num_qubits': self.num_qubits}

        # General case
        if front:
            if other.output_dims() != self.input_dims(qargs):
                raise QiskitError(
                    "Other operator output dimensions ({}) does not"
                    " match current input dimensions ({}).".format(
                        other.output_dims(qargs), self.input_dims()))
            output_dims = self.output_dims()
            if qargs is None:
                input_dims = other.input_dims()
            else:
                input_dims = list(self.input_dims())
                for qubit, dim in zip(qargs, other.input_dims()):
                    input_dims[qubit] = dim
        else:
            if other.input_dims() != self.output_dims(qargs):
                raise QiskitError(
                    "Other operator input dimensions ({}) does not"
                    " match current output dimensions ({}).".format(
                        other.output_dims(qargs), self.input_dims()))
            input_dims = self.input_dims()
            if qargs is None:
                output_dims = other.output_dims()
            else:
                output_dims = list(self.output_dims())
                for qubit, dim in zip(qargs, other.output_dims()):
                    output_dims[qubit] = dim
        return {'input_dims': input_dims,
                'output_dims': output_dims,
                'num_qubits': None}

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented

        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (Matrix, Number, )):
                return NotImplemented

        inputs = tuple(i._array if isinstance(i, Matrix)
                       else i for i in inputs)
        if out:
            kwargs['out'] = tuple(i._array if isinstance(i, Matrix)
                                  else i for i in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if isinstance(result, tuple):
            # multiple return values
            return tuple(type(self)(i) for i in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return type(self)(result)

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, (Matrix, ) + self._HANDLED_TYPES) for t in types):
            return NotImplemented

        if func in self._HANDLED_FUNCTIONS:
            return self._HANDLED_FUNCTIONS[func](*args, **kwargs)

        if func in self._DELEGATE_FUNCTIONS:
            args = tuple(i._array if isinstance(i, Matrix) else i for i in args)
            return func(*args, **kwargs)

        return NotImplemented
