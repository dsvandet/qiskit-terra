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
Qiskit singledispatch overloads for Numpy
"""

# Import numpy namespace
from numpy import *

# Override select numpy functions with singledispatch versions
from .numpy_dispatch import *

# Include dispatch definitions for other backends
from .numpy_jax import _HAS_JAX, JaxParam, JaxParam
from .numpy_tensorflow import _HAS_TENSORFLOW, TensorflowArray

# Array conversion for supported array backends
from .asarray import asarray, is_array, array_backends
