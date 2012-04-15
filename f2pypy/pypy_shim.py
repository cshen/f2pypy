# Numpypy doesn't implement enough of the environment for f2py to
# work. Add what it's mising.

import sys
import numpy
from ctypes import *

if not hasattr(numpy, "intp"):
    n = sizeof(POINTER(c_int))
    if n == sizeof(c_int32):
        def intp():
            return numpy.dtype("int32")
    elif n == sizeof(c_int64):
        def intp():
            return numpy.dtype("int32")
    else:
        raise AssertionError

    numpy.intp = intp

try:
    import numpy.testing
except ImportError:
    from . import pypy_testing_shim
    sys.modules["numpy.testing"] = pypy_testing_shim
