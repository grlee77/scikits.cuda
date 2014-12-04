#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

"""
Python interface to cuSPARSE functions.

Note: You may need to set the environment variable CUDA_ROOT to the base of
your CUDA installation.
"""
# import low level cuSPARSE python wrappers and constants

try:
    from ._cusparse_cffi import *
except Exception as e:
    estr = "autogenerattion and import of cuSPARSE wrappers failed\n"
    estr += ("Try setting the CUDA_ROOT environment variable to the base of"
             "your CUDA installation.  The autogeneration script tries to find"
             "the CUSPARSE header at CUDA_ROOT/include/cusparse_v2.h\n")
    raise ImportError(estr)

# define higher level wrappers for common functions
# will check dimensions, autoset some variables and call the appriopriate
# function based on the input dtype

def dense_nnz(handle, descrA, A, dirA=CUSPARSE_DIRECTION_ROW, lda=None,
        nnzPerRowCol=None, nnzTotalDevHostPtr=None):
    """ higher level wrapper to cusparse<t>nnz routines """
    if not isinstance(A, pycuda.gpuarray.GPUArray):
        raise ValueError("A must be a pyCUDA gpuarray")
    if lda is None:
        lda = A.shape[0]

    m, n = A.shape
    assert lda >= m
    dtype = A.dtype

    if nnzPerRowCol is None:
        if dirA == CUSPARSE_DIRECTION_ROW:
            nnzPerRowCol = gpuarray.to_gpu(np.zeros((m, ), dtype=np.int32))
        elif dirA == CUSPARSE_DIRECTION_COLUMN:
            nnzPerRowCol = gpuarray.to_gpu(np.zeros((n, ), dtype=np.int32))
        else:
            raise ValueError("Invalid dirA")
    if nnzTotalDevHostPtr is None:
        nnzTotalDevHostPtr = ffi.new('int *', 0)
    if dtype == np.float32:
        fn = cusparseSnnz
    elif dtype == np.float64:
        fn = cusparseDnnz
    elif dtype == np.complex64:
        fn = cusparseCnnz
    elif dtype == np.complex128:
        fn = cusparseZnnz
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)
    fn(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol,
       nnzTotalDevHostPtr)
    return nnzPerRowCol, nnzTotalDevHostPtr[0]


def csrmv(handle, descrA, csrValA, csrRowPtrA, csrColIndA, m, n, x, nnz=None,
          transA=CUSPARSE_OPERATION_NON_TRANSPOSE, alpha=1.0, beta=0.0,
          y=None, check_inputs=True):
    """ higher level wrapper to cusparse<t>csrmv routines """

    if check_inputs:
        if not isinstance(csrValA, pycuda.gpuarray.GPUArray):
            raise ValueError("csrValA must be a pyCUDA gpuarray")
        if not isinstance(csrRowPtrA, pycuda.gpuarray.GPUArray):
            raise ValueError("csrRowPtrA must be a pyCUDA gpuarray")
        if not isinstance(csrColIndA, pycuda.gpuarray.GPUArray):
            raise ValueError("csrColIndA must be a pyCUDA gpuarray")
        if not isinstance(x, pycuda.gpuarray.GPUArray):
            raise ValueError("x must be a pyCUDA gpuarray")

    if nnz is None:
        nnz = csrValA.size

    dtype = csrValA.dtype
    if y is None:
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            y = gpuarray.zeros((m, ), dtype=dtype)
        else:
            y = gpuarray.zeros((n, ), dtype=dtype)

    # perform some basic sanity checks
    if check_inputs:
        if csrValA.size != nnz:
            raise ValueError("length of csrValA array must match nnz")

        if (x.dtype != dtype) or (y.dtype != dtype):
            raise ValueError("incompatible dtypes")

        if csrRowPtrA.size != (m+1):
            raise ValueError("length of csrRowPtrA array must be m+1")

        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            if x.size != n:
                raise ValueError("sizes of x, A incompatible")
            if y.size != m:
                raise ValueError("sizes of y, A incompatible")
        else:
            if x.size != m:
                raise ValueError("sizes of x, A incompatible")
            if y.size != n:
                raise ValueError("sizes of y, A incompatible")

    if dtype == np.float32:
        fn = cusparseScsrmv
    elif dtype == np.float64:
        fn = cusparseDcsrmv
    elif dtype == np.complex64:
        fn = cusparseCcsrmv
    elif dtype == np.complex128:
        fn = cusparseZcsrmv
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)
    fn(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA,
       csrColIndA, x, beta, y)
    return y