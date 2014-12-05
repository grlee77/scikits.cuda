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
    if len(A.shape) != 2:
        raise ValueError("A must be 2D")
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


def dense2csr(A, handle=None, descrA=None, lda=None, check_inputs=True):

    # try moving list or numpy array to GPU
    if not isinstance(A, pycuda.gpuarray.GPUArray):
        A = np.atleast_2d(A)
        A = gpuarray.to_gpu(A)

    if check_inputs:
        if not isinstance(A, pycuda.gpuarray.GPUArray):
            raise ValueError("A must be a pyCUDA gpuarray")
        if len(A.shape) != 2:
            raise ValueError("A must be 2D")
        if descrA is not None:
            if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
                raise ValueError("Only general matrix type supported")

    if lda is None:
        lda = A.shape[0]
    m, n = A.shape
    assert lda >= m
    dtype = A.dtype

    if handle is None:
        handle = cusparseCreate()
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    if descrA is None:
        descrA = cusparseCreateMatDescr()
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    nnzPerRow, nnz = dense_nnz(
        handle, descrA, A, dirA=CUSPARSE_DIRECTION_ROW, lda=lda)

    csrRowPtrA = gpuarray.to_gpu(np.zeros((m+1, ), dtype=np.int32))
    csrColIndA = gpuarray.to_gpu(np.zeros((nnz, ), dtype=np.int32))
    csrValA = gpuarray.to_gpu(np.zeros((nnz, ), dtype=dtype))

    if dtype == np.float32:
        fn = cusparseSdense2csr
    elif dtype == np.float64:
        fn = cusparseDdense2csr
    elif dtype == np.complex64:
        fn = cusparseCdense2csr
    elif dtype == np.complex128:
        fn = cusparseZdense2csr
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)

    fn(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA,
       csrColIndA)

    return (handle, descrA, csrValA, csrRowPtrA, csrColIndA)


def csr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A=None,
              lda=None, check_inputs=True):

    if check_inputs:
        if A is not None:
            if not isinstance(A, pycuda.gpuarray.GPUArray):
                raise ValueError("A must be a pyCUDA gpuarray")
            if len(A.shape) != 2:
                raise ValueError("A must be 2D")
        if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
        if cusparseGetMatIndexBase(descrA) != CUSPARSE_INDEX_BASE_ZERO:
            raise ValueError("Only base 0 matrix supported")
        for arr in [csrValA, csrRowPtrA, csrColIndA]:
            if not isinstance(arr, pycuda.gpuarray.GPUArray):
                raise ValueError("csr* inputs must be a pyCUDA gpuarrays")
        if (csrValA.size != m) or (csrColIndA.size != m):
            raise ValueError("A: inconsistent size")
        if (csrRowPtrA.size != m + 1):
            raise ValueError("A: inconsistent size")

    if lda is None:
        lda = m
    assert lda >= m

    dtype = csrValA.dtype
    A = gpuarray.to_gpu(np.zeros((m, n), dtype=dtype))

    if dtype == np.float32:
        fn = cusparseScsr2dense
    elif dtype == np.float64:
        fn = cusparseDcsr2dense
    elif dtype == np.complex64:
        fn = cusparseCcsr2dense
    elif dtype == np.complex128:
        fn = cusparseZcsr2dense
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)

    fn(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda)

    return A


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


def csrmm(handle, m, n, k, descrA, csrValA, csrRowPtrA, csrColIndA, B, C=None,
          nnz=None, transA=CUSPARSE_OPERATION_NON_TRANSPOSE, alpha=1.0,
          beta=0.0, ldb=None, ldc=None, check_inputs=True):
    """ higher level wrapper to cusparse<t>csrmm routines """

    if check_inputs:
        for item in [csrValA, csrRowPtrA, csrColIndA, B]:
            if not isinstance(item, pycuda.gpuarray.GPUArray):
                raise ValueError("csr*, B, must be pyCUDA gpuarrays")
        if C is not None:
            if not isinstance(C, pycuda.gpuarray.GPUArray):
                raise ValueError("C must be a pyCUDA gpuarray or None")
        # dense matrices must be in column-major order
        if not B.flags.f_contiguous:
            raise ValueError("Dense matrix B must be in column-major order")

    dtype = csrValA.dtype

    if C is None:
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ldc = m
        else:
            ldc = k
        C = gpuarray.zeros((ldc, n), dtype=dtype, order='F')
    elif not C.flags.f_contiguous:
        raise ValueError("Dense matrix C must be in column-major order")

    if nnz is None:
        nnz = csrValA.size

    if ldb is None:
        ldb = B.shape[0]

    if ldc is None:
        ldc = C.shape[0]

    # perform some basic sanity checks
    if check_inputs:
        if csrValA.size != nnz:
            raise ValueError("length of csrValA array must match nnz")

        if (B.dtype != dtype) or (C.dtype != dtype):
            raise ValueError("A, B, C must share a common dtype")

        if ldb < B.shape[0]:
            raise ValueError("ldb invalid for matrix B")

        if ldc < C.shape[0]:
            raise ValueError("ldc invalid for matrix C")

        if (C.shape[1] != n) or (B.shape[1] != n):
            raise ValueError("bad shape for B or C")

        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            if (ldb != k) or (ldc != m):
                raise ValueError("size of A incompatible with B or C")
        else:
            if (ldb != m) or (ldc != k):
                raise ValueError("size of A incompatible with B or C")

        if csrRowPtrA.size != m+1:
            raise ValueError("length of csrRowPtrA invalid")

    if dtype == np.float32:
        fn = cusparseScsrmm
    elif dtype == np.float64:
        fn = cusparseDcsrmm
    elif dtype == np.complex64:
        fn = cusparseCcsrmm
    elif dtype == np.complex128:
        fn = cusparseZcsrmm
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)
    fn(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
       csrColIndA, B, ldb, beta, C, ldc)
    return C


def csrmm2(handle, m, n, k, descrA, csrValA, csrRowPtrA, csrColIndA, B, C=None,
           nnz=None, transA=CUSPARSE_OPERATION_NON_TRANSPOSE,
           transB=CUSPARSE_OPERATION_NON_TRANSPOSE, alpha=1.0,
           beta=0.0, ldb=None, ldc=None, check_inputs=True):
    """ higher level wrapper to cusparse<t>csrmm routines """

    if check_inputs:
        for item in [csrValA, csrRowPtrA, csrColIndA, B]:
            if not isinstance(item, pycuda.gpuarray.GPUArray):
                raise ValueError("csr*, B, must be pyCUDA gpuarrays")
        if C is not None:
            if not isinstance(C, pycuda.gpuarray.GPUArray):
                raise ValueError("C must be a pyCUDA gpuarray or None")
        # dense matrices must be in column-major order
        if not B.flags.f_contiguous:
            raise ValueError("Dense matrix B must be in column-major order")

        if transB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
            raise ValueError("Conjugate transpose operation not supported for"
                             "dense matrix B")

        if (transB == CUSPARSE_OPERATION_TRANSPOSE) and \
           (transA != CUSPARSE_OPERATION_NON_TRANSPOSE):
               raise ValueError("if B is transposed, only A non-transpose"
                                "is supported")

    dtype = csrValA.dtype

    if C is None:
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ldc = m
        else:
            ldc = k
        C = gpuarray.zeros((ldc, n), dtype=dtype, order='F')
    elif not C.flags.f_contiguous:
        raise ValueError("Dense matrix C must be in column-major order")

    if nnz is None:
        nnz = csrValA.size

    if ldb is None:
        ldb = B.shape[0]

    if ldc is None:
        ldc = C.shape[0]

    # perform some basic sanity checks
    if check_inputs:
        if csrValA.size != nnz:
            raise ValueError("length of csrValA array must match nnz")

        if (B.dtype != dtype) or (C.dtype != dtype):
            raise ValueError("A, B, C must share a common dtype")

        if ldb < B.shape[0]:
            raise ValueError("ldb invalid for matrix B")

        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ldOpA=m  # leading dimension for op(A)
            tdOpA=k  # trailing dimension for op(A)
        else:
            ldOpA=k
            tdOpA=m

        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
            if B.shape[1] != n:
                raise ValueError("B, n incompatible")
            if (ldb < tdOpA):
                raise ValueError("size of A incompatible with B")
        else:
            if ldb < n:
                raise ValueError("B, n incompatible")
            if (B.shape[1] != tdOpA):
                raise ValueError("size of A incompatible with B")

        if (C.shape[1] != n):
            raise ValueError("bad shape for C")

        if (ldc != ldOpA):
            raise ValueError("size of A incompatible with C")

        if csrRowPtrA.size != m+1:
            raise ValueError("length of csrRowPtrA invalid")

    if dtype == np.float32:
        fn = cusparseScsrmm2
    elif dtype == np.float64:
        fn = cusparseDcsrmm2
    elif dtype == np.complex64:
        fn = cusparseCcsrmm2
    elif dtype == np.complex128:
        fn = cusparseZcsrmm2
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)
    transa = transA
    transb = transB
    try:
        fn(handle, transa, transb, m, n, k, nnz, alpha, descrA, csrValA,
           csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc)
    except CUSPARSE_STATUS_INVALID_VALUE as e:
        print("m={}, n={}, k={}, nnz={}, ldb={}, ldc={}".format(
            m, n, k, nnz, ldb, ldc))
        raise(e)
    return C


def _csrgemmNnz(handle, m, n, k, descrA, csrRowPtrA, csrColIndA, descrB,
            csrRowPtrB, csrColIndB, descrC, csrRowPtrC, nnzA=None, nnzB=None,
            transA=CUSPARSE_OPERATION_NON_TRANSPOSE,
            transB=CUSPARSE_OPERATION_NON_TRANSPOSE,
            check_inputs=True):
    """ higher level wrapper to cusparseXcsrgemmNnz.

    Note
    ----
    transA(A) is shape m x k.  transB(B) is shape k x n.  C is shape m x n
    """
    if check_inputs:
        for array in [csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB]:
            if not isinstance(array, pycuda.gpuarray.GPUArray):
                raise ValueError("all csr* inputs must be a pyCUDA gpuarray")
        if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
        if cusparseGetMatType(descrB) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
    if nnzA is None:
        nnzA = csrColIndA.size
    if nnzB is None:
        nnzB = csrColIndB.size

    nnzTotalDevHostPtr = ffi.new('int *', 0)

    # perform some basic sanity checks
    if check_inputs:
        if csrColIndA.size != nnzA:
            raise ValueError("length of csrValA array must match nnzA")
        if csrColIndB.size != nnzB:
            raise ValueError("length of csrValB array must match nnzB")
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ptrA_size = m + 1
        else:
            ptrA_size = k + 1
        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ptrB_size = k + 1
        else:
            ptrB_size = n + 1
        if csrRowPtrA.size != ptrA_size:
            raise ValueError("length of csrRowPtrA array must be m+1")
        if csrRowPtrB.size != ptrB_size:
            raise ValueError("length of csrRowPtrB array must be n+1")

    cusparseXcsrgemmNnz(handle, transA, transB, m, n, k, descrA, nnzA,
                        csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB,
                        csrColIndB, descrC, csrRowPtrC, nnzTotalDevHostPtr)
    nnzC = nnzTotalDevHostPtr[0]
    return nnzC


def csrgemm(handle, m, n, k, descrA, csrValA, csrRowPtrA, csrColIndA, descrB,
            csrValB, csrRowPtrB, csrColIndB, nnzA=None, nnzB=None,
            transA=CUSPARSE_OPERATION_NON_TRANSPOSE,
            transB=CUSPARSE_OPERATION_NON_TRANSPOSE,
            check_inputs=True):
    """ higher level wrapper to cusparse<t>csrgemm routines.

    Note
    ----
    transA(A) is shape m x k.  transB(B) is shape k x n.  C is shape m x n

    if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
        m, k = A.shape
    else:
        k, m = A.shape

    if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
        k, n = B.shape
    else:
        n, k = B.shape

    """

    if check_inputs:
        for array in [csrValA, csrRowPtrA, csrColIndA, csrValB, csrRowPtrB,
                      csrColIndB]:
            if not isinstance(array, pycuda.gpuarray.GPUArray):
                raise ValueError("all csr* inputs must be a pyCUDA gpuarray")
        if cusparseGetMatType(descrA) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
        if cusparseGetMatType(descrB) != CUSPARSE_MATRIX_TYPE_GENERAL:
            raise ValueError("Only general matrix type supported")
    if nnzA is None:
        nnzA = csrValA.size
    if nnzB is None:
        nnzB = csrValB.size

    dtype = csrValA.dtype

    # perform some basic sanity checks
    if check_inputs:
        if csrValA.size != nnzA:
            raise ValueError("length of csrValA array must match nnzA")
        if csrValB.size != nnzB:
            raise ValueError("length of csrValB array must match nnzB")

        if (dtype != csrValB.dtype):
            raise ValueError("incompatible dtypes")

    # perform some basic sanity checks
    if check_inputs:
        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ptrA_size = m + 1
        else:
            ptrA_size = k + 1
        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
            ptrB_size = k + 1
        else:
            ptrB_size = n + 1
        if csrRowPtrA.size != ptrA_size:
            raise ValueError("bad csrRowPtrA size")
        if csrRowPtrB.size != ptrB_size:
            raise ValueError("bad csrRowPtrB size")

    # allocate output matrix C descr and row pointers
    descrC = cusparseCreateMatDescr()
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL)
    csrRowPtrC = gpuarray.to_gpu(np.zeros((m+1, ), dtype=np.int32))

    # call csrgemmNnz to determine nnzC and fill in csrRowPtrC
    nnzC = _csrgemmNnz(handle, m, n, k, descrA, csrRowPtrA, csrColIndA,
                       descrB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC,
                       nnzA=nnzA, nnzB=nnzB, transA=transA, transB=transB,
                       check_inputs=False)

    # allocated rest of C based on nnzC
    csrValC = gpuarray.to_gpu(np.zeros((nnzC, ), dtype=dtype))
    csrColIndC = gpuarray.to_gpu(np.zeros((nnzC, ), dtype=np.int32))

    if dtype == np.float32:
        fn = cusparseScsrgemm
    elif dtype == np.float64:
        fn = cusparseDcsrgemm
    elif dtype == np.complex64:
        fn = cusparseCcsrgemm
    elif dtype == np.complex128:
        fn = cusparseZcsrgemm
    else:
        raise ValueError("unsupported sparse matrix dtype: %s" % dtype)

    fn(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA,
       csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC,
       csrValC, csrRowPtrC, csrColIndC)
    return (descrC, csrValC, csrRowPtrC, csrColIndC)



class CSR:
    def __init__(self, handle, descr, csrVal, csrRowPtr, csrColInd, shape):
        self.handle = handle
        self.descr = descr
        self.Val = csrVal
        self.RowPtr = csrRowPtr
        self.ColInd = csrColInd
        self.nnz = csrVal.size

        # mirror scipys.sparse names
        self.data = csrVal
        self.indices = csrColInd
        self.indptr = csrRowPtr

        # TODO: change these to properties?
        if descr is not None:
            self.matrix_type = cusparseGetMatType(descr)
            self.index_base = cusparseGetMatIndexBase(descr)
            self.diag_type = cusparseGetMatDiagType(descr)
            self.fill_mode = cusparseGetMatFillMode(descr)
        if csrVal is not None:
            self.dtype = csrVal.dtype

        self.shape = shape

    # alternative constructor from dense ndarray, gpuarray or cuSPARSE matrix
    @classmethod
    def to_CSR(cls, A, handle=None):
        """Takes dense numpy array and returns in CSR format"""
        (handle, descr, csrVal, csrRowPtr, csrColInd) = dense2csr(A, handle)
        return cls(handle, descr, csrVal, csrRowPtr, csrColInd, A.shape)

    def to_dense(self, lda = None):
        """ returns dense gpuarray A """
        m, n = self.shape
        if lda is None:
            lda = m
        else:
            assert lda >= m
        A = csr2dense(self.handle, m, n, self.descr, self.Val, self.RowPtr,
            self.ColInd, lda=lda)
        return A

    # TODO: overload __mul__, etc.

    def __repr__(self):
        rstr = "CSR matrix:\n"
        rstr += "\tshape = {}\n".format(self.shape)
        rstr += "\tdtype = {}\n".format(self.dtype)
        rstr += "\tMatrixType = {}\n".format(self.matrix_type)
        rstr += "\tIndexBase = {}\n".format(self.index_base)
        rstr += "\tDiagType = {}\n".format(self.diag_type)
        rstr += "\tFillMode = {}\n".format(self.fill_mode)
        rstr += "\tcontext = {}\n\n".format(self.handle)
        rstr += "\tnnz = {}\n".format(self.nnz)
        rstr += "\tRowPtr = {}\n".format(self.RowPtr)
        rstr += "\tVal = {}\n".format(self.Val)
        return rstr
