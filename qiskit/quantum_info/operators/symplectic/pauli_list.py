# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Optimized list of Pauli operators
"""

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.symplectic.pauli_table import PauliTable
from qiskit.quantum_info.operators.symplectic.stabilizer_table import StabilizerTable
from qiskit.quantum_info.operators.mixins import LinearMixin, GroupMixin


class PauliList(BasePauli, LinearMixin, GroupMixin):
    r"""List of N-qubit Pauli operators.

    This class is an efficient representation of a list of
    :class:`Pauli` operators. It supports 1D numpy array indexing
    returning a :class:`Pauli` for integer indexes or a
    :class:`PauliList` for slice or list indices.

    **Initialization**

    A PauliList object can be initialized in several ways.

        ``PauliList(list[str])``
            where strings are same representation with :class:`~qiskit.quantum_info.Pauli`.

        ``PauliList(Pauli) and PauliList(list[Pauli])``
            where Pauli is :class:`~qiskit.quantum_info.Pauli`.

        ``PauliList.from_symplectic(z, x, phase)``
            where ``z`` and ``x`` are 2 dimensional boolean ``numpy.ndarrays`` and ``phase`` is
            an integer in ``[0, 1, 2, 3]``.

    For example,

    .. jupyter-execute::

        import numpy as np

        from qiskit.quantum_info import Pauli, PauliList

        # 1. init from list[str]
        pauli_list = PauliList(["II", "+ZI", "-iYY"])
        print("1. ", pauli_list)

        pauli1 = Pauli("iXI")
        pauli2 = Pauli("iZZ")

        # 2. init from Pauli
        print("2. ", PauliList(pauli1))

        # 3. init from list[Pauli]
        print("3. ", PauliList([pauli1, pauli2]))

        # 4. init from np.ndarray
        z = np.array([[True, True], [False, False]])
        x = np.array([[False, True], [True, False]])
        phase = np.array([0, 1])
        pauli_list = PauliList.from_symplectic(z, x)
        print("4. ", pauli_list)

    **Data Access**

    The individual Paulis can be accessed and updated using the ``[]``
    operator which accepts integer, lists, or slices for selecting subsets
    of PauliList. If integer is given, it returns Pauli not PauliList.

    .. jupyter-execute::

        pauli_list = PauliList(["XX", "ZZ", "IZ"])
        print("Integer: ", repr(pauli_list[1]))
        print("List: ", repr(pauli_list[[0, 2]]))
        print("Slice: ", repr(pauli_list[0:2]))

    **Iteration**

    Rows in the Pauli table can be iterated over like a list. Iteration can
    also be done using the label or matrix representation of each row using the
    :meth:`label_iter` and :meth:`matrix_iter` methods.
    """

    # Set the max number of qubits * paulis before string truncation
    __truncate__ = 2000

    def __init__(self, data=None, *, z=None, x=None, phase=None, id=None, method=None):
        """Initialize the PauliList.

        Args:
            data (Pauli or list): input data for Paulis. If input is a list each item in the list
                                  must be a Pauli object or Pauli str.

        Args:
            data ([type], optional): [description]. Defaults to None.
            z ([type], optional): [description]. Defaults to None.
            x ([type], optional): [description]. Defaults to None.
            phases ([type], optional): [description]. Defaults to None.
            method ([type], optional): [description]. Defaults to None. 

        Examples:

            method=None input types
            =======================
            PauliList()

            PauliList(<tuple>) : Tuple is (z,x,phases) or (z,x)
            PauliList(<PauliList>)
            PauliList(<PauliTable>)
            PauliList(<StabilizerTable>)
            PauliList(<list of Paulis>)
            PauliList(<list of Pauli strings>)
            PauliList(<BasePauli>/<Pauli>)
            PauliList(<np.ndarray of Paulis>)
            PauliList(<np.ndarary of Pauli strings>)
            ...

            PauliList(x=np.ndarray, z=np.ndarray, phase=np.ndarray)

            method='lookup' input types
            =======================
            PauliList(id=str, method='lookup')

            method='gui' input types
            =======================
            PauliList(method='gui')       

        Raises:
            QiskitError: if input array is invalid shape.

        Additional Information:
            The input array is not copied so multiple Pauli tables
            can share the same underlying array.
        """
        self.empty = False
        if method is None:
            if data is None and z is None:
                # Generate an empty PauliList
                self.empty = True
            elif data is None:
                # Conversion from keyword input
                if x is not None:
                    if phase is None:
                        array_z, array_x, array_phase = self.from_array(z, x)
                    else:
                        array_z, array_x, array_phase = self.from_array(z, x, phase)
                else:
                    raise QiskitError("Both x and z must be provided")
            elif isinstance(data, BasePauli):
                # TODO: add a internal_phase property to BasePauli so that we do not have
                # to access the private variable _phase
                array_z, array_x, array_phase = data.z, data.x, data._phase
            elif isinstance(data, tuple):
                if len(data) not in [2, 3]:
                    raise QiskitError(
                        "Invalid input tuple for Pauli, input tuple must be"
                        " `(z, x, phase)` or `(z, x)`"
                    )
                array_z, array_x, array_phase = self._from_array(*data)
            elif isinstance(data, StabilizerTable):
                # Conversion from legacy StabilizerTable
                array_z, array_x, array_phase = self._from_array(data.Z, data.X, 2 * data.phase)
            elif isinstance(data, PauliTable):
                # Conversion from legacy PauliTable
                array_z, array_x, array_phase = self._from_array(data.Z, data.X)
            else:
                # Conversion as iterable of Paulis
                array_z, array_x, array_phase = self._from_paulis(data)
        elif method == 'lookup':
            if id is None:
                raise QiskitError("The PauliList id missing.")
            else:
                raise NotImplementedError
        elif method == 'gui':
            raise NotImplementedError
        else:
            raise QiskitError("Unknown or unsupported input")

        # Initialize BasePauli
        if self.empty is not True:
            super().__init__(array_z, array_x, array_phase)
        else:
            self.num_paulis = 0
            self.num_qubits = 0

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    def __array__(self, dtype=None):
        """Custome numpy array container: Convert to numpy array"""
        # pylint: disable=unused-argument
        shape = (len(self),) + 2 * (2 ** self.num_qubits,)
        result = np.zeros(shape, dtype=complex)
        for i, mat in enumerate(self.matrix_iter()):
            result[i] = mat
        return result

    @staticmethod
    def _from_paulis(data):
        """Construct a PauliList from a list of Pauli data.
        
        Args:
            data (iterable): list of Pauli data.

        Returns:
            PauliList: the constructed PauliList.

        Raises:
            QiskitError: If the input list is empty or contains invalid
            Pauli strings.
        """
        if not isinstance(data, (list, tuple, np.ndarray, set)):
            data = [data]
        num_paulis = len(data)
        if num_paulis == 0:
            return None, None, None
        paulis = []
        for i in data:
            if not isinstance(i, Pauli):
                paulis.append(Pauli(i))
            else:
                paulis.append(i)
        num_qubits = paulis[0].num_qubits
        array_z = np.zeros((num_paulis, num_qubits), dtype=bool)
        array_x = np.zeros((num_paulis, num_qubits), dtype=bool)
        array_phase = np.zeros(num_paulis, dtype=np.int8)
        for i, pauli in enumerate(paulis):
            array_z[i] = pauli._z
            array_x[i] = pauli._x
            array_phase[i] = pauli._phase
        return array_z, array_x, array_phase

    def __repr__(self):
        """Display representation."""
        return self._truncated_str(True)

    def __str__(self):
        """Print representation."""
        return self._truncated_str(False)

    def _truncated_str(self, show_class):
        """Truncate string representations if desired"""
        stop = self._num_paulis
        if self.__truncate__:
            max_paulis = self.__truncate__ // self.num_qubits
            if self._num_paulis > max_paulis:
                stop = max_paulis
        labels = [str(self[i]) for i in range(stop)]
        prefix = "PauliList(" if show_class else ""
        tail = ")" if show_class else ""
        if stop != self._num_paulis:
            suffix = ", ...]" + tail
        else:
            suffix = "]" + tail
        list_str = np.array2string(
            np.array(labels), threshold=stop + 1, separator=", ", prefix=prefix, suffix=suffix
        )
        return prefix + list_str[:-1] + suffix

    # ---------------------------------------------------------------------
    # Comparisions
    # ---------------------------------------------------------------------

    def __eq__(self, other):
        """Entrywise comparison of Pauli equality."""
        if not isinstance(other, PauliList):
            other = PauliList(other)
        if not isinstance(other, BasePauli):
            return False
        return self._eq(other)

    def equiv(self, other):
        """Entrywise comparison of Pauli equivalence up to global phase.

        Args:
            other (PauliList or Pauli): a comparison object.

        Returns:
            np.ndarray: An array of True or False for entrywise equivalence
                        of the current table.
        """
        if not isinstance(other, PauliList):
            other = PauliList(other)
        return np.all(self.z == other.z, axis=1) & np.all(self.x == other.x, axis=1)

    # ---------------------------------------------------------------------
    # Direct array access
    # ---------------------------------------------------------------------
    @property
    def phase(self):
        """Return the external phase exponents of the PauliList."""
        # Convert the internal phase expoents to external phase exponents
        phase = self.change_representation(
            self._phase,
            y_count = self._count_y(self.x, self.z),
        )

        return phase

    @phase.setter
    def phase(self, value):
        """Set the phase exponents of the PauliList. Input phase
        exponents should be in __external_pauli_rep_format__"""
        # Convert external phase exponents to the internal phases exponents
        self._phase[:] = self.change_representation(
            value,
            y_count = self._count_y(self.x, self.z)
        )

    @property
    def coeff(self):
        """Return the actual complex phase coefficients of a PauliList 
        - not phase exponents"""
        phase = self.convert_phase_exponent(
            self._phase, 
            self.internal_pauli_format(),
            self.external_pauli_format())
        return self._exponent_to_coeff(phase, self.external_pauli_format())

    @coeff.setter
    def coeff(self, value):
        """Set the complex phase coefficients for the PauliList"""
        if all(coeff not in [1,0-1j,-1,1j] for coeff in value):
            raise QiskitError("Coefficient must be one of 1, -1, i, -i")
        phase = self._coeff_to_exponent(value, self.external_phase_format)
        self._phase[:] = self.convert_phase_exponent(
            phase,
            self.external_pauli_format,
            self.internal_pauli_format)

    @property
    def order(self):
        """Return the group orders of the elements of the PauliList"""
        return self._order()

    @property
    def x(self):
        """The X array for the symplectic representation."""
        return self._x

    @x.setter
    def x(self, val):
        self._x[:] = val

    @property
    def z(self):
        """The Z array for the symplectic representation."""
        return self._z

    @z.setter
    def z(self, val):
        self._z[:] = val

    @property
    def y(self):
        """The Y array form the symplectic representation."""
        return np.logical_and(self._z, self._x)

    @property
    def phase_rep_format(self):
        """Return the current external phase format."""
        return self.external_phase_format

    @phase_rep_format.setter
    def phase_rep_format(self, rep_format):
        """Set the external phase format"""
        self.set_formats(phase_format=rep_format)

    @property
    def symp_rep_format(self):
        """Return the external symplectic format"""
        return self.external_symp_format

    @symp_rep_format.setter
    def symp_rep_format(self, rep_format):
        """Set the external symplectic representation format"""
        self.set_formats(symp_format=rep_format)

    @property
    def pauli_rep_format(self):
        return self.external_pauli_format

    @pauli_rep_format.setter
    def pauli_rep_format(self, rep_format):
        """Set the external pauli representation format"""
        if rep_format not in self.phase_formats:
            raise QiskitError("Not a supported Pauli representation")
        phase_format, symp_format = self._split_rep(rep_format)
        self.set_formats(phase_format=phase_format, symp_format=symp_format)

    # ---------------------------------------------------------------------
    # Size Properties
    # ---------------------------------------------------------------------

    @property
    def shape(self):
        """The full shape of the :meth:`array`"""
        return self._num_paulis, self.num_qubits

    @property
    def size(self):
        """The number of Pauli rows in the table."""
        return self._num_paulis

    def __len__(self):
        """Return the number of Pauli rows in the table."""
        return self._num_paulis

    # ---------------------------------------------------------------------
    # Pauli Array methods
    # ---------------------------------------------------------------------

    def __getitem__(self, index):
        """Return a view of the PauliList."""
        # Returns a view of specified rows of the PauliList
        # This supports all slicing operations the underlying array supports.
        if isinstance(index, tuple):
            if len(index) == 1:
                index = index[0]
            elif len(index) > 2:
                raise IndexError("Invalid PauliList index {}".format(index))

        # Row-only indexing
        if isinstance(index, (int, np.integer)):
            # Single Pauli
            return Pauli(
                BasePauli(
                    self._z[np.newaxis, index],
                    self._x[np.newaxis, index],
                    self._phase[np.newaxis, index],
                )
            )
        elif isinstance(index, (slice, list, np.ndarray)):
            # Sub-Table view
            return PauliList(BasePauli(self._z[index], self._x[index], self._phase[index]))

        # Row and Qubit indexing
        return PauliList((self._z[index], self._x[index], 0))

    def __setitem__(self, index, value):
        """Update PauliList."""
        if isinstance(index, tuple):
            if len(index) == 1:
                index = index[0]
            elif len(index) > 2:
                raise IndexError("Invalid PauliList index {}".format(index))

        # Modify specified rows of the PauliList
        if not isinstance(value, PauliList):
            value = PauliList(value)

        self._z[index] = value._z
        self._x[index] = value._x
        if not isinstance(index, tuple):
            # Row-only indexing
            self._phase[index] = value._phase
        else:
            # Row and Qubit indexing
            self._phase[index[0]] += value._phase
            self._phase %= 4

    def delete(self, ind, qubit=False):
        """Return a copy with Pauli rows deleted from table.

        When deleting qubits the qubit index is the same as the
        column index of the underlying :attr:`X` and :attr:`Z` arrays.

        Args:
            ind (int or list): index(es) to delete.
            qubit (bool): if True delete qubit columns, otherwise delete
                          Pauli rows (Default: False).

        Returns:
            PauliList: the resulting table with the entries removed.

        Raises:
            QiskitError: if ind is out of bounds for the array size or
                         number of qubits.
        """
        if isinstance(ind, int):
            ind = [ind]

        # Row deletion
        if not qubit:
            if max(ind) >= len(self):
                raise QiskitError(
                    "Indices {} are not all less than the size"
                    " of the PauliList ({})".format(ind, len(self))
                )
            z = np.delete(self._z, ind, axis=0)
            x = np.delete(self._x, ind, axis=0)
            phase = np.delete(self._phase, ind)

            return PauliList(BasePauli(z, x, phase))

        # Column (qubit) deletion
        if max(ind) >= self.num_qubits:
            raise QiskitError(
                "Indices {} are not all less than the number of"
                " qubits in the PauliList ({})".format(ind, self.num_qubits)
            )
        z = np.delete(self._z, ind, axis=1)
        x = np.delete(self._x, ind, axis=1)
        # Use self.phase, not self._phase as deleting qubits can change the
        # ZX phase convention
        return PauliList.from_symplectic(z, x, self.phase)

    def insert(self, ind, value, qubit=False):
        """Insert Pauli's into the table.

        When inserting qubits the qubit index is the same as the
        column index of the underlying :attr:`X` and :attr:`Z` arrays.

        Args:
            ind (int): index to insert at.
            value (PauliList): values to insert.
            qubit (bool): if True delete qubit columns, otherwise delete
                          Pauli rows (Default: False).

        Returns:
            PauliList: the resulting table with the entries inserted.

        Raises:
            QiskitError: if the insertion index is invalid.
        """
        if not isinstance(ind, int):
            raise QiskitError("Insert index must be an integer.")

        if not isinstance(value, PauliList):
            value = PauliList(value)

        # Row insertion
        size = self._num_paulis
        if not qubit:
            if ind > size:
                raise QiskitError(
                    "Index {} is larger than the number of rows in the"
                    " PauliList ({}).".format(ind, size)
                )
            base_z = np.insert(self._z, ind, value._z, axis=0)
            base_x = np.insert(self._x, ind, value._x, axis=0)
            base_phase = np.insert(self._phase, ind, value._phase)
            return PauliList(BasePauli(base_z, base_x, base_phase))

        # Column insertion
        if ind > self.num_qubits:
            raise QiskitError(
                "Index {} is greater than number of qubits"
                " in the PauliList ({})".format(ind, self.num_qubits)
            )
        # TODO: Fix phase
        if len(value) == 1:
            # Pad blocks to correct size
            value_x = np.vstack(size * [value.x])
            value_z = np.vstack(size * [value.z])
        elif len(value) == size:
            #  Blocks are already correct size
            value_x = value.x
            value_z = value.z
        else:
            # Blocks are incorrect size
            raise QiskitError(
                "Input PauliList must have a single row, or"
                " the same number of rows as the Pauli Table"
                " ({}).".format(size)
            )
        # Build new array by blocks
        z = np.hstack([self.z[:, :ind], value_z, self.z[:, ind:]])
        x = np.hstack([self.x[:, :ind], value_x, self.x[:, ind:]])

        return PauliList.from_symplectic(z, x, self.phase)

    def argsort(self, weight=False, phase=False):
        """Return indices for sorting the rows of the table.

        The default sort method is lexicographic sorting by qubit number.
        By using the `weight` kwarg the output can additionally be sorted
        by the number of non-identity terms in the Pauli, where the set of
        all Pauli's of a given weight are still ordered lexicographically.

        Args:
            weight (bool): Optionally sort by weight if True (Default: False).
            phase (bool): Optionally sort by phase before weight or order
                          (Default: False).

        Returns:
            array: the indices for sorting the table.
        """
        # Get order of each Pauli using
        # I => 0, X => 1, Y => 2, Z => 3
        x = self.x
        z = self.z
        order = 1 * (x & ~z) + 2 * (x & z) + 3 * (~x & z)
        phases = self._phase
        # Optionally get the weight of Pauli
        # This is the number of non identity terms
        if weight:
            weights = np.sum(x | z, axis=1)

        # To preserve ordering between successive sorts we
        # are use the 'stable' sort method
        indices = np.arange(self._num_paulis)

        # Initial sort by phases
        sort_inds = phases.argsort(kind="stable")
        indices = indices[sort_inds]
        order = order[sort_inds]
        if phase:
            phases = phases[sort_inds]
        if weight:
            weights = weights[sort_inds]

        # Sort by order
        for i in range(self.num_qubits):
            sort_inds = order[:, i].argsort(kind="stable")
            order = order[sort_inds]
            indices = indices[sort_inds]
            if weight:
                weights = weights[sort_inds]
            if phase:
                phases = phases[sort_inds]

        # If using weights we implement a sort by total number
        # of non-identity Paulis
        if weight:
            indices = indices[weights.argsort(kind="stable")]

        # If sorting by phase we perform a final sort by the phase value
        # of each pauli
        if phase:
            indices = indices[phases.argsort(kind="stable")]
        return indices

    def sort(self, weight=False, phase=False):
        """Sort the rows of the table.

        The default sort method is lexicographic sorting by qubit number.
        By using the `weight` kwarg the output can additionally be sorted
        by the number of non-identity terms in the Pauli, where the set of
        all Pauli's of a given weight are still ordered lexicographically.

        **Example**

        Consider sorting all a random ordering of all 2-qubit Paulis

        .. jupyter-execute::

            from numpy.random import shuffle
            from qiskit.quantum_info.operators import PauliList

            # 2-qubit labels
            labels = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
                      'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
            # Shuffle Labels
            shuffle(labels)
            pt = PauliList(labels)
            print('Initial Ordering')
            print(pt)

            # Lexicographic Ordering
            srt = pt.sort()
            print('Lexicographically sorted')
            print(srt)

            # Weight Ordering
            srt = pt.sort(weight=True)
            print('Weight sorted')
            print(srt)

        Args:
            weight (bool): optionally sort by weight if True (Default: False).
            phase (bool): Optionally sort by phase before weight or order
                          (Default: False).

        Returns:
            PauliList: a sorted copy of the original table.
        """
        return self[self.argsort(weight=weight, phase=phase)]

    def unique(self, return_index=False, return_counts=False):
        """Return unique Paulis from the table.

        **Example**

        .. jupyter-execute::

            from qiskit.quantum_info.operators import PauliList

            pt = PauliList(['X', 'Y', '-X', 'I', 'I', 'Z', 'X', 'iZ'])
            unique = pt.unique()
            print(unique)

        Args:
            return_index (bool): If True, also return the indices that
                                 result in the unique array.
                                 (Default: False)
            return_counts (bool): If True, also return the number of times
                                  each unique item appears in the table.

        Returns:
            PauliList: unique
                the table of the unique rows.

            unique_indices: np.ndarray, optional
                The indices of the first occurrences of the unique values in
                the original array. Only provided if ``return_index`` is True.

            unique_counts: np.array, optional
                The number of times each of the unique values comes up in the
                original array. Only provided if ``return_counts`` is True.
        """
        # Check if we need to stack the phase array
        if np.any(self._phase != self._phase[0]):
            # Create a single array of Pauli's and phases for calling np.unique on
            # so that we treat different phased Pauli's as unique
            array = np.hstack([self._z, self._x, self.phase.reshape((self.phase.shape[0], 1))])
        else:
            # All Pauli's have the same phase so we only need to sort the array
            array = np.hstack([self._z, self._x])

        # Get indexes of unique entries
        if return_counts:
            _, index, counts = np.unique(array, return_index=True, return_counts=True, axis=0)
        else:
            _, index = np.unique(array, return_index=True, axis=0)

        # Sort the index so we return unique rows in the original array order
        sort_inds = index.argsort()
        index = index[sort_inds]
        unique = PauliList(BasePauli(self._z[index], self._x[index], self._phase[index]))

        # Concatinate return tuples
        ret = (unique,)
        if return_index:
            ret += (index,)
        if return_counts:
            ret += (counts[sort_inds],)
        if len(ret) == 1:
            return ret[0]
        return ret

    def _order(self):
        _IENC = {1:1, -1:2, 1j:4, 0-1j:4}
        _NENC = {1:2, -1:2, 1j:4, 0-1j:4}
        phase = self.convert_phase_exponent(
            self._phase, 
            self.__INTERNAL_PAULI_REP_FROMAT__, 
            'iYZX')
        order = np.zeros(self.num_paulis, dtype=np.int8)
        identities_z = np.asarray([(np.all(rowz==0) for rowz in self._z)])
        identities_x = np.asarray([(np.all(rowx==0) for rowx in self._x)])
        identities = np.logical_and(identities_z, identities_x).nonzero()[0]
        identities = identities.nonzero()[0]
        for r in range(self.num_paulis):
            if r in identities:
                order[r]=_IENC[phase][r]
            else:
                order[r]=_NENC[phase][r]
        return order

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def tensor(self, other):
        """Return the tensor product with each Pauli in the list.

        Args:
            other (PauliList): another PauliList.

        Returns:
            PauliList: the list of tensor product Paulis.

        Raises:
            QiskitError: if other cannot be converted to a PauliList, does
                         not have either 1 or the same number of Paulis as
                         the current list.
        """
        if not isinstance(other, PauliList):
            other = PauliList(other)
        if len(other) not in [1, len(self)]:
            raise QiskitError(
                "Incompatible PauliLists. Other list must "
                "have either 1 or the same number of Paulis."
            )
        return PauliList(super().tensor(other))

    def expand(self, other):
        """Return the expand product of each Pauli in the list.

        Args:
            other (PauliList): another PauliList.

        Returns:
            PauliList: the list of tensor product Paulis.

        Raises:
            QiskitError: if other cannot be converted to a PauliList, does
                         not have either 1 or the same number of Paulis as
                         the current list.
        """
        if not isinstance(other, PauliList):
            other = PauliList(other)
        if len(other) not in [1, len(self)]:
            raise QiskitError(
                "Incompatible PauliLists. Other list must "
                "have either 1 or the same number of Paulis."
            )
        return PauliList(super().expand(other))

    def compose(self, other, qargs=None, front=False, inplace=False):
        """Return the composition self∘other for each Pauli in the list.

        Args:
            other (PauliList): another PauliList.
            qargs (None or list): qubits to apply dot product on (Default: None).
            front (bool): If True use `dot` composition method [default: False].
            inplace (bool): If True update in-place (default: False).

        Returns:
            PauliList: the list of composed Paulis.

        Raises:
            QiskitError: if other cannot be converted to a PauliList, does
                         not have either 1 or the same number of Paulis as
                         the current list, or has the wrong number of qubits
                         for the specified qargs.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, PauliList):
            other = PauliList(other)
        if len(other) not in [1, len(self)]:
            raise QiskitError(
                "Incompatible PauliLists. Other list must "
                "have either 1 or the same number of Paulis."
            )
        return PauliList(super().compose(other, qargs=qargs, front=front, inplace=inplace))

    # pylint: disable=arguments-differ
    def dot(self, other, qargs=None, inplace=False):
        """Return the composition other∘self for each Pauli in the list.

        Args:
            other (PauliList): another PauliList.
            qargs (None or list): qubits to apply dot product on (Default: None).
            inplace (bool): If True update in-place (default: False).

        Returns:
            PauliList: the list of composed Paulis.

        Raises:
            QiskitError: if other cannot be converted to a PauliList, does
                         not have either 1 or the same number of Paulis as
                         the current list, or has the wrong number of qubits
                         for the specified qargs.
        """
        return self.compose(other, qargs=qargs, front=True, inplace=inplace)

    def _add(self, other, qargs=None):
        """Append two PauliLists.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (PauliList): another table.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            PauliList: the concatinated list self + other.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, PauliList):
            other = PauliList(other)

        self._op_shape._validate_add(other._op_shape, qargs)

        base_phase = np.hstack((self._phase, other._phase))

        if qargs is None or (sorted(qargs) == qargs and len(qargs) == self.num_qubits):
            base_z = np.vstack([self._z, other._z])
            base_x = np.vstack([self._x, other._x])
        else:
            # Pad other with identity and then add
            padded = BasePauli(
                np.zeros((self.size, self.num_qubits), dtype=bool),
                np.zeros((self.size, self.num_qubits), dtype=bool),
                np.zeros(self.size, dtype=int),
            )
            padded = padded.compose(other, qargs=qargs, inplace=True)
            base_z = np.vstack([self._z, padded._z])
            base_x = np.vstack([self._x, padded._x])

        return PauliList(BasePauli(base_z, base_x, base_phase))

    def _multiply(self, other):
        """Multiply each Pauli in the list by a phase.

        Args:
            other (complex or array): a complex number in [1, -1j, -1, 1j]

        Returns:
            PauliList: the list of Paulis other * self.

        Raises:
            QiskitError: if the phase is not in the set [1, -1j, -1, 1j].
        """
        return PauliList(super()._multiply(other))

    def conjugate(self):
        """Return the conjugate of each Pauli in the list."""
        return PauliList(super().conjugate())

    def transpose(self):
        """Return the transpose of each Pauli in the list."""
        return PauliList(super().transpose())

    def adjoint(self):
        """Return the adjoint of each Pauli in the list."""
        return PauliList(super().adjoint())

    def inverse(self):
        """Return the inverse of each Pauli in the list."""
        return PauliList(super().adjoint())

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------

    def commutes(self, other, qargs=None):
        """Return True for each Pauli that commutes with other.

        Args:
            other (PauliList): another PauliList operator.
            qargs (list): qubits to apply dot product on (default: None).

        Returns:
            bool: True if Pauli's commute, False if they anti-commute.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, BasePauli):
            other = PauliList(other)
        return super().commutes(other, qargs=qargs)

    def anticommutes(self, other, qargs=None):
        """Return True if other Pauli that anticommutes with other.

        Args:
            other (PauliList): another PauliList operator.
            qargs (list): qubits to apply dot product on (default: None).

        Returns:
            bool: True if Pauli's anticommute, False if they commute.
        """
        return np.logical_not(self.commutes(other, qargs=qargs))

    def commutes_with_all(self, other):
        """Return indexes of rows that commute other.

        If other is a multi-row Pauli list the returned vector indexes rows
        of the current PauliList that commute with *all* Pauli's in other.
        If no rows satisfy the condition the returned array will be empty.

        Args:
            other (PauliList): a single Pauli or multi-row PauliList.

        Returns:
            array: index array of the commuting rows.
        """
        return self._commutes_with_all(other)

    def anticommutes_with_all(self, other):
        """Return indexes of rows that commute other.

        If other is a multi-row Pauli list the returned vector indexes rows
        of the current PauliList that anti-commute with *all* Pauli's in other.
        If no rows satisfy the condition the returned array will be empty.

        Args:
            other (PauliList): a single Pauli or multi-row PauliList.

        Returns:
            array: index array of the anti-commuting rows.
        """
        return self._commutes_with_all(other, anti=True)

    def _commutes_with_all(self, other, anti=False):
        """Return row indexes that commute with all rows in another PauliList.

        Args:
            other (PauliList): a PauliList.
            anti (bool): if True return rows that anti-commute, otherwise
                         return rows that commute (Default: False).

        Returns:
            array: index array of commuting or anti-commuting row.
        """
        if not isinstance(other, PauliList):
            other = PauliList(other)
        comms = self.commutes(other[0])
        (inds,) = np.where(comms == int(not anti))
        for pauli in other[1:]:
            comms = self[inds].commutes(pauli)
            (new_inds,) = np.where(comms == int(not anti))
            if new_inds.size == 0:
                # No commuting rows
                return new_inds
            inds = inds[new_inds]
        return inds

    def evolve(self, other, qargs=None):
        r"""Evolve the Pauli by a Clifford.

        This returns the Pauli :math:`P^\prime = C.P.C^\dagger`.

        Args:
            other (Pauli or Clifford or QuantumCircuit): The Clifford operator to evolve by.
            qargs (list): a list of qubits to apply the Clifford to.

        Returns:
            Pauli: the Pauli :math:`C.P.C^\dagger`.

        Raises:
            QiskitError: if the Clifford number of qubits and qargs don't match.
        """
        from qiskit.circuit import Instruction, QuantumCircuit
        from qiskit.quantum_info.operators.symplectic.clifford import Clifford

        if qargs is None:
            qargs = getattr(other, "qargs", None)

        # Convert quantum circuits to Cliffords
        if isinstance(other, Clifford):
            other = other.to_circuit()

        if not isinstance(other, (BasePauli, Instruction, QuantumCircuit)):
            # Convert to a PauliList
            other = PauliList(other)

        return PauliList(super().evolve(other, qargs=qargs))

    def to_labels(self, array=False):
        r"""Convert a PauliList to a list Pauli string labels.

        For large PauliLists converting using the ``array=True``
        kwarg will be more efficient since it allocates memory for
        the full Numpy array of labels in advance.

        .. list-table:: Pauli Representations
            :header-rows: 1

            * - Label
              - Symplectic
              - Matrix
            * - ``"I"``
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
            * - ``"X"``
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
            * - ``"Y"``
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & -i \\ i & 0  \end{bmatrix}`
            * - ``"Z"``
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`

        Args:
            array (bool): return a Numpy array if True, otherwise
                          return a list (Default: False).

        Returns:
            list or array: The rows of the PauliList in label form.
        """
        ret = np.zeros(self.size, dtype="<U{}".format(self.num_qubits))
        iterator = self.label_iter()
        for i in range(self.size):
            ret[i] = next(iterator)
        if array:
            return ret
        return ret.tolist()

    def to_matrix(self, sparse=False, array=False):
        r"""Convert to a list or array of Pauli matrices.

        For large PauliLists converting using the ``array=True``
        kwarg will be more efficient since it allocates memory a full
        rank-3 Numpy array of matrices in advance.

        .. list-table:: Pauli Representations
            :header-rows: 1

            * - Label
              - Symplectic
              - Matrix
            * - ``"I"``
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
            * - ``"X"``
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
            * - ``"Y"``
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & -i \\ i & 0  \end{bmatrix}`
            * - ``"Z"``
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`

        Args:
            sparse (bool): if True return sparse CSR matrices, otherwise
                           return dense Numpy arrays (Default: False).
            array (bool): return as rank-3 numpy array if True, otherwise
                          return a list of Numpy arrays (Default: False).

        Returns:
            list: A list of dense Pauli matrices if `array=False` and `sparse=False`.
            list: A list of sparse Pauli matrices if `array=False` and `sparse=True`.
            array: A dense rank-3 array of Pauli matrices if `array=True`.
        """
        if not array:
            # We return a list of Numpy array matrices
            return list(self.matrix_iter(sparse=sparse))
        # For efficiency we also allow returning a single rank-3
        # array where first index is the Pauli row, and second two
        # indices are the matrix indices
        dim = 2 ** self.num_qubits
        ret = np.zeros((self.size, dim, dim), dtype=complex)
        iterator = self.matrix_iter(sparse=sparse)
        for i in range(self.size):
            ret[i] = next(iterator)
        return ret

    # ---------------------------------------------------------------------
    # Custom Iterators
    # ---------------------------------------------------------------------

    def label_iter(self):
        """Return a label representation iterator.

        This is a lazy iterator that converts each row into the string
        label only as it is used. To convert the entire table to labels use
        the :meth:`to_labels` method.

        Returns:
            LabelIterator: label iterator object for the PauliList.
        """

        class LabelIterator(CustomIterator):
            """Label representation iteration and item access."""

            def __repr__(self):
                return "<PauliList_label_iterator at {}>".format(hex(id(self)))

            def __getitem__(self, key):
                return self.obj._to_label(self.obj._z[key], self.obj._x[key], self.obj._phase[key])

        return LabelIterator(self)

    def matrix_iter(self, sparse=False):
        """Return a matrix representation iterator.

        This is a lazy iterator that converts each row into the Pauli matrix
        representation only as it is used. To convert the entire table to
        matrices use the :meth:`to_matrix` method.

        Args:
            sparse (bool): optionally return sparse CSR matrices if True,
                           otherwise return Numpy array matrices
                           (Default: False)

        Returns:
            MatrixIterator: matrix iterator object for the PauliList.
        """

        class MatrixIterator(CustomIterator):
            """Matrix representation iteration and item access."""

            def __repr__(self):
                return "<PauliList_matrix_iterator at {}>".format(hex(id(self)))

            def __getitem__(self, key):
                return self.obj._to_matrix(
                    self.obj._z[key], self.obj._x[key], self.obj._phase[key], sparse=sparse
                )

        return MatrixIterator(self)

    # ---------------------------------------------------------------------
    # Class methods
    # ---------------------------------------------------------------------

    @classmethod
    def from_symplectic(cls, z, x, phase=0):
        """Construct a PauliList from a symplectic data.

        Args:
            z (np.ndarray): 2D boolean Numpy array.
            x (np.ndarray): 2D boolean Numpy array.
            phase (np.ndarray or None): Optional, 1D integer array from Z_4.

        Returns:
            PauliList: the constructed PauliList.
        """
        base_z, base_x, base_phase = cls._from_array(z, x, phase)
        return cls(BasePauli(base_z, base_x, base_phase))
