#!/usr/bin/env python
#
# Created by: Pearu Peterson, April 2002
#

__usage__ = """
Build linalg:
  python setup.py build
Run tests if scipy is installed:
  python -c 'import scipy;scipy.linalg.test()'
"""

import math

# The original code uses numpy's testing environment.
# I reimplemented just enough of it to pass most of the tests
# using CPython.
from f2pypy import pypy_shim

import numpy as np
from f2pypy.testing import TestCase, run_module_suite, assert_equal, \
    assert_almost_equal, assert_array_almost_equal

#from scipy.linalg import fblas, cblas, get_blas_funcs
#import blas as fblas
#import blas as cblas
### Change to use the f2pypy generated version
# (which doesn't support the cblas interface).
import fblas
cblas = fblas
# I found that I got confused about which environment I was in
print "Using", fblas

_type_conv = {'f':'s', 'd':'d', 'F':'c', 'D':'z', 'G':'z'}
_blas_alias = {'cnrm2' : 'scnrm2', 'znrm2' : 'dznrm2',
               'cdot' : 'cdotc', 'zdot' : 'zdotc',
               'cger' : 'cgerc', 'zger' : 'zgerc',
               'sdotc': 'sdot', 'sdotu': 'sdot',
               'ddotc': 'ddot', 'ddotu': 'ddot'}


def get_blas_funcs(names, arrays=(), dtype=None):
    blas_funcs = []
    unpack = False
    dtype = np.dtype(dtype)
    module1 = (cblas, 'cblas')
    module2 = (fblas, 'fblas')

    if isinstance(names, str):
        names = (names,)
        unpack = True

    if arrays:
        # use the most generic type in arrays
        dtype, index = max(
            [(ar.dtype, i) for i, ar in enumerate(arrays)])
        if arrays[index].flags['FORTRAN']:
            # prefer Fortran for leading array with column major order
            module1, module2 = module2, module1

    prefix = _type_conv.get(dtype.char, 'd')

    for i, name in enumerate(names):
        func_name = prefix + name
        func_name = _blas_alias.get(func_name, func_name)
        func = getattr(module1[0], func_name, None)
        module_name = module1[1]
        if func is None:
            func = getattr(module2[0], func_name, None)
            module_name = module2[1]
        if func is None:
            raise ValueError(
                'BLAS function %s could not be found' % func_name)
        func.module_name, func.typecode = module_name, prefix
        blas_funcs.append(func)

    if unpack:
        return blas_funcs[0]
    else:
        return blas_funcs


def test_get_blas_funcs():
    # check that it returns Fortran code for arrays that are
    # fortran-ordered
    f1, f2, f3 = get_blas_funcs(
        ('axpy', 'axpy', 'axpy'),
        (np.empty((2,2), dtype=np.complex64, order='F'),
         np.empty((2,2), dtype=np.complex128, order='C'))
        )

    # get_blas_funcs will choose libraries depending on most generic
    # array
    assert_equal(f1.typecode, 'z')
    assert_equal(f1.module_name, 'cblas')
    assert_equal(f2.typecode, 'z')
    assert_equal(f2.module_name, 'cblas')

    # check defaults.
    f1 = get_blas_funcs('rotg')
    assert_equal(f1.typecode, 'd')

    # check also dtype interface
    f1 = get_blas_funcs('gemm', dtype=np.complex64)
    assert_equal(f1.typecode, 'c')
    f1 = get_blas_funcs('gemm', dtype='F')
    assert_equal(f1.typecode, 'c')

    # extended precision complex
    f1 = get_blas_funcs('gemm', dtype=np.longcomplex)
    assert_equal(f1.typecode, 'z')

def test_get_blas_funcs_alias():
    # check alias for get_blas_funcs
    f, g = get_blas_funcs(('nrm2', 'dot'), dtype=np.complex64)
    assert f.typecode == 'c'
    assert g.typecode == 'c'

    f, g, h = get_blas_funcs(('dot', 'dotc', 'dotu'), dtype=np.float64)
    assert f is g
    assert f is h

class TestCBLAS1Simple(TestCase):

    def test_axpy(self):
        for p in 'sd':
            f = getattr(cblas,p+'axpy',None)
            if f is None: continue
            assert_array_almost_equal(f(5,[1,2,3],[2,-1,3]),[7,9,18])
        for p in 'cz':
            f = getattr(cblas,p+'axpy',None)
            if f is None: continue
            assert_array_almost_equal(f(5,[1,2j,3],[2,-1,3]),[7,10j-1,18])

class TestFBLAS1Simple(TestCase):

    def test_axpy(self):
        for p in 'sd':
            f = getattr(fblas,p+'axpy',None)
            if f is None: continue
            assert_array_almost_equal(f([1,2,3],[2,-1,3],a=5),[7,9,18])
        for p in 'cz':
            f = getattr(fblas,p+'axpy',None)
            if f is None: continue
            assert_array_almost_equal(f([1,2j,3],[2,-1,3],a=5),[7,10j-1,18])

    def test_copy(self):
        for p in 'sd':
            f = getattr(fblas,p+'copy',None)
            if f is None: continue
            assert_array_almost_equal(f([3,4,5],[8]*3),[3,4,5])
        for p in 'cz':
            f = getattr(fblas,p+'copy',None)
            if f is None: continue
            assert_array_almost_equal(f([3,4j,5+3j],[8]*3),[3,4j,5+3j])

    def test_asum(self):
        for p in 'sd':
            f = getattr(fblas,p+'asum',None)
            if f is None: continue
            assert_almost_equal(f([3,-4,5]),12)
        for p in ['sc','dz']:
            f = getattr(fblas,p+'asum',None)
            if f is None: continue
            assert_almost_equal(f([3j,-4,3-4j]),14)

    def test_dot(self):
        for p in 'sd':
            f = getattr(fblas,p+'dot',None)
            if f is None: continue
            assert_almost_equal(f([3,-4,5],[2,5,1]),-9)

    def test_complex_dotu(self):
        for p in 'cz':
            f = getattr(fblas,p+'dotu',None)
            if f is None: continue
            assert_almost_equal(f([3j,-4,3-4j],[2,3,1]),-9+2j)

    def test_complex_dotc(self):
        for p in 'cz':
            f = getattr(fblas,p+'dotc',None)
            if f is None: continue
            assert_almost_equal(f([3j,-4,3-4j],[2,3j,1]),3-14j)

    def test_nrm2(self):
        for p in 'sd':
            f = getattr(fblas,p+'nrm2',None)
            if f is None: continue
            assert_almost_equal(f([3,-4,5]),math.sqrt(50))
        for p in ['c', 'z', 'sc','dz']:
            f = getattr(fblas,p+'nrm2',None)
            if f is None: continue
            assert_almost_equal(f([3j,-4,3-4j]),math.sqrt(50))

    def test_scal(self):
        for p in 'sd':
            f = getattr(fblas,p+'scal',None)
            if f is None: continue
            assert_array_almost_equal(f(2,[3,-4,5]),[6,-8,10])
        for p in 'cz':
            f = getattr(fblas,p+'scal',None)
            if f is None: continue
            assert_array_almost_equal(f(3j,[3j,-4,3-4j]),[-9,-12j,12+9j])
        for p in ['cs','zd']:
            f = getattr(fblas,p+'scal',None)
            if f is None: continue
            assert_array_almost_equal(f(3,[3j,-4,3-4j]),[9j,-12,9-12j])

    def test_swap(self):
        for p in 'sd':
            f = getattr(fblas,p+'swap',None)
            if f is None: continue
            x,y = [2,3,1],[-2,3,7]
            x1,y1 = f(x,y)
            assert_array_almost_equal(x1,y)
            assert_array_almost_equal(y1,x)
        for p in 'cz':
            f = getattr(fblas,p+'swap',None)
            if f is None: continue
            x,y = [2,3j,1],[-2,3,7-3j]
            x1,y1 = f(x,y)
            assert_array_almost_equal(x1,y)
            assert_array_almost_equal(y1,x)

    def test_amax(self):
        for p in 'sd':
            f = getattr(fblas,'i'+p+'amax')
            assert_equal(f([-2,4,3]),1)
        for p in 'cz':
            f = getattr(fblas,'i'+p+'amax')
            assert_equal(f([-5,4+3j,6]),1)
    #XXX: need tests for rot,rotm,rotg,rotmg

class TestFBLAS2Simple(TestCase):

    def test_gemv(self):
        for p in 'sd':
            f = getattr(fblas,p+'gemv',None)
            if f is None: continue
            assert_array_almost_equal(f(3,[[3]],[-4]),[-36])
            assert_array_almost_equal(f(3,[[3]],[-4],3,[5]),[-21])
        for p in 'cz':
            f = getattr(fblas,p+'gemv',None)
            if f is None: continue
            assert_array_almost_equal(f(3j,[[3-4j]],[-4]),[-48-36j])
            assert_array_almost_equal(f(3j,[[3-4j]],[-4],3,[5j]),[-48-21j])

    def test_ger(self):

        for p in 'sd':
            f = getattr(fblas,p+'ger',None)
            if f is None: continue
            assert_array_almost_equal(f(1,[1,
                                           2],[3,4]),[[3,4],[6,8]])
            assert_array_almost_equal(f(2,[1,
                                           2,
                                           3],[3,4]),[[6,8],[12,16],[18,24]])

            assert_array_almost_equal(f(1,[1,
                                           2],[3,4],
                                        a=[[1,2],[3,4]]
                                        ),[[4,6],[9,12]])

        for p in 'cz':
            f = getattr(fblas,p+'geru',None)
            if f is None: continue
            assert_array_almost_equal(f(1,[1j,
                                           2],[3,4]),[[3j,4j],[6,8]])
            assert_array_almost_equal(f(-2,[1j,
                                           2j,
                                           3j],[3j,4j]),[[6,8],[12,16],[18,24]])

        for p in 'cz':
            for name in ('ger', 'gerc'):
                f = getattr(fblas,p+name,None)
                if f is None: continue
                assert_array_almost_equal(f(1,[1j,
                                               2],[3,4]),[[3j,4j],[6,8]])
                assert_array_almost_equal(f(2,[1j,
                                               2j,
                                               3j],[3j,4j]),[[6,8],[12,16],[18,24]])

class TestFBLAS3Simple(TestCase):

    def test_gemm(self):
        for p in 'sd':
            f = getattr(fblas,p+'gemm',None)
            if f is None: continue
            assert_array_almost_equal(f(3,[3],[-4]),[[-36]])
            assert_array_almost_equal(f(3,[3],[-4],3,[5]),[-21])
        for p in 'cz':
            f = getattr(fblas,p+'gemm',None)
            if f is None: continue
            assert_array_almost_equal(f(3j,[3-4j],[-4]),[[-48-36j]])
            assert_array_almost_equal(f(3j,[3-4j],[-4],3,[5j]),[-48-21j])

if __name__ == "__main__":
    run_module_suite()
