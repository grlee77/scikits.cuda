#!/usr/bin/env python

"""
Python interface to CUSPARSE functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import atexit
import ctypes.util
import platform
from string import Template
import sys
import warnings

import numpy as np

import cuda

# Load library:
_version_list = [6.5, 6.0, 5.5, 5.0, 4.0]
if 'linux' in sys.platform:
    _libcusparse_libname_list = ['libcusparse.so'] + \
                                ['libcusparse.so.%s' % v for v in _version_list]
elif sys.platform == 'darwin':
    _libcusparse_libname_list = ['libcusparse.dylib']
elif sys.platform == 'win32':
    if platform.machine().endswith('64'):
        _libcusparse_libname_list = ['cusparse.dll'] + \
                                    ['cusparse64_%s.dll' % int(10*v) for v in _version_list]
    else:
        _libcusparse_libname_list = ['cusparse.dll'] + \
                                    ['cusparse32_%s.dll' % int(10*v) for v in _version_list]
else:
    raise RuntimeError('unsupported platform')

# Print understandable error message when library cannot be found:
_libcusparse = None
for _libcusparse_libname in _libcusparse_libname_list:
    try:
        if sys.platform == 'win32':
            _libcusparse = ctypes.windll.LoadLibrary(_libcusparse_libname)
        else:
            _libcusparse = ctypes.cdll.LoadLibrary(_libcusparse_libname)
    except OSError:
        pass
    else:
        break
if _libcusparse == None:
    OSError('CUDA sparse library not found')

class cusparseError(Exception):
    """CUSPARSE error"""
    pass

class cusparseStatusNotInitialized(cusparseError):
    """CUSPARSE library not initialized"""
    pass

class cusparseStatusAllocFailed(cusparseError):
    """CUSPARSE resource allocation failed"""
    pass

class cusparseStatusInvalidValue(cusparseError):
    """Unsupported value passed to the function"""
    pass

class cusparseStatusArchMismatch(cusparseError):
    """Function requires a feature absent from the device architecture"""
    pass

class cusparseStatusMappingError(cusparseError):
    """An access to GPU memory space failed"""
    pass

class cusparseStatusExecutionFailed(cusparseError):
    """GPU program failed to execute"""
    pass

class cusparseStatusInternalError(cusparseError):
    """An internal CUSPARSE operation failed"""
    pass

class cusparseStatusMatrixTypeNotSupported(cusparseError):
    """The matrix type is not supported by this function"""
    pass

cusparseExceptions = {
    1: cusparseStatusNotInitialized,
    2: cusparseStatusAllocFailed,
    3: cusparseStatusInvalidValue,
    4: cusparseStatusArchMismatch,
    5: cusparseStatusMappingError,
    6: cusparseStatusExecutionFailed,
    7: cusparseStatusInternalError,
    8: cusparseStatusMatrixTypeNotSupported,
    }

# Matrix types:
CUSPARSE_MATRIX_TYPE_GENERAL = 0
CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1
CUSPARSE_MATRIX_TYPE_HERMITIAN = 2
CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3

# indicate whether upper or lower part is stored:
CUSPARSE_FILL_MODE_LOWER = 0
CUSPARSE_FILL_MODE_UPPER = 1

# Whether or not a matrix' diagonal entries are unity:
CUSPARSE_DIAG_TYPE_NON_UNIT = 0
CUSPARSE_DIAG_TYPE_UNIT = 1

# Matrix index bases:
CUSPARSE_INDEX_BASE_ZERO = 0
CUSPARSE_INDEX_BASE_ONE = 1

# Operation types:
CUSPARSE_OPERATION_NON_TRANSPOSE = 0
CUSPARSE_OPERATION_TRANSPOSE = 1
CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2

# Whether or not to parse elements of a dense matrix row or column-wise.
CUSPARSE_DIRECTION_ROW = 0
CUSPARSE_DIRECTION_COLUMN = 1

# operation is performed on indices only or on both data and indices
CUSPARSE_ACTION_SYMBOLIC = 0
CUSPARSE_ACTION_NUMERIC = 1

# how to perform partitioning of a HYB matrix into ELL and COO
CUSPARSE_HYB_PARTITION_AUTO = 0    
CUSPARSE_HYB_PARTITION_USER = 1 
CUSPARSE_HYB_PARTITION_MAX = 2

# scalar values passed by reference on the host or device
CUSPARSE_POINTER_MODE_HOST = 0
CUSPARSE_POINTER_MODE_DEVICE = 1

# whether to generate and use level information in csrsv2, csric02, csrilu02,
# bsrsv2, bsric02 and bsrilu02
CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0
CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1

# Helper functions:
class cusparseMatDescr(ctypes.Structure):
    _fields_ = [
        ('MatrixType', ctypes.c_int),
        ('FillMode', ctypes.c_int),
        ('DiagType', ctypes.c_int),
        ('IndexBase', ctypes.c_int)
        ]

def cusparseCheckStatus(status):
    """
    Raise CUSPARSE exception

    Raise an exception corresponding to the specified CUSPARSE error
    code.

    Parameters
    ----------
    status : int
        CUSPARSE error code.

    See Also
    --------
    cusparseExceptions

    """

    if status != 0:
        try:
            raise cusparseExceptions[status]
        except KeyError:
            raise cusparseError

_libcusparse.cusparseCreate.restype = int
_libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]
def cusparseCreate():
    """
    Initialize CUSPARSE.

    Initializes CUSPARSE and creates a handle to a structure holding
    the CUSPARSE library context.

    Returns
    -------
    handle : int
        CUSPARSE library context.

    """

    handle = ctypes.c_int()
    status = _libcusparse.cusparseCreate(ctypes.byref(handle))
    cusparseCheckStatus(status)
    return handle.value

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_int]
def cusparseDestroy(handle):
    """
    Release CUSPARSE resources.

    Releases hardware resources used by CUSPARSE

    Parameters
    ----------
    handle : int
        CUSPARSE library context.

    """

    status = _libcusparse.cusparseDestroy(handle)
    cusparseCheckStatus(status)

_libcusparse.cusparseCreateHybMat.restype = int
_libcusparse.cusparseCreateHybMat.argtypes = [ctypes.c_void_p]
def cusparseCreateHybMat():
    """
    Initialize opaque CUSPARSE Hybrid Matrix hybA data structure.

    Returns
    -------
    hybA : int
        CUSPARSE library context.

    """

    hybA = ctypes.c_int()
    status = _libcusparse.cusparseCreateHybMat(ctypes.byref(hybA))
    cusparseCheckStatus(status)
    return hybA.value

_libcusparse.cusparseDestroyHybMat.restype = int
_libcusparse.cusparseDestroyHybMat.argtypes = [ctypes.c_int]
def cusparseDestroyHybMat(hybA):
    """
    Release CUSPARSE Hybrid matrix resources.

    Releases hardware resources used by CUSPARSE Hybrid matrix

    Parameters
    ----------
    hybA : int
        CUSPARSE hybrid format storage structure.

    """

    status = _libcusparse.cusparseDestroyHybMat(hybA)
    cusparseCheckStatus(status)

_libcusparse.cusparseCreateSolveAnalysisInfo.restype = int
_libcusparse.cusparseCreateSolveAnalysisInfo.argtypes = [ctypes.c_void_p]
def cusparseCreateSolveAnalysisInfo():
    """
    Initializes the opaque solve and analysis structure to default values.

    Returns
    -------
    hybA : int
        CUSPARSE library context.

    """

    info = ctypes.c_int()
    status = _libcusparse.cusparseCreateSolveAnalysisInfo(ctypes.byref(info))
    cusparseCheckStatus(status)
    return info.value

_libcusparse.cusparseDestroySolveAnalysisInfo.restype = int
_libcusparse.cusparseDestroySolveAnalysisInfo.argtypes = [ctypes.c_int]
def cusparseDestroySolveAnalysisInfo(info):
    """
    Release CUSPARSE resources.

    Releases hardware resources used by CUSPARSE analysis structure

    Parameters
    ----------
    info : int
        CUSPARSE analysis structure.

    """

    status = _libcusparse.cusparseDestroySolveAnalysisInfo(info)
    cusparseCheckStatus(status)

_libcusparse.cusparseGetVersion.restype = int
_libcusparse.cusparseGetVersion.argtypes = [ctypes.c_int,
                                            ctypes.c_void_p]
def cusparseGetVersion(handle):
    """
    Return CUSPARSE library version.

    Returns the version number of the CUSPARSE library.

    Parameters
    ----------
    handle : int
        CUSPARSE library context.

    Returns
    -------
    version : int
        CUSPARSE library version number.

    """

    version = ctypes.c_int()
    status = _libcusparse.cusparseGetVersion(handle,
                                             ctypes.byref(version))
    cusparseCheckStatus(status)
    return version.value

_libcusparse.cusparseSetStream.restype = int
_libcusparse.cusparseSetStream.argtypes = [ctypes.c_int,
                                                 ctypes.c_int]
def cusparseSetStream(handle, id):
    """
    Sets the CUSPARSE stream in which kernels will run.

    Parameters
    ----------
    handle : int
        CUSPARSE library context.
    id : int
        Stream ID.

    """

    status = _libcusparse.cusparseSetStream(handle, id)
    cusparseCheckStatus(status)

_libcusparse.cusparseCreateMatDescr.restype = int
_libcusparse.cusparseCreateMatDescr.argtypes = [cusparseMatDescr]
def cusparseCreateMatDescr():
    """
    Initialize a sparse matrix descriptor.

    Initializes the `MatrixType` and `IndexBase` fields of the matrix
    descriptor to the default values `CUSPARSE_MATRIX_TYPE_GENERAL`
    and `CUSPARSE_INDEX_BASE_ZERO`.

    Returns
    -------
    desc : cusparseMatDescr
        Matrix descriptor.

    """

    desc = cusparseMatDescr()
    status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(desc))
    cusparseCheckStatus(status)
    return desc

_libcusparse.cusparseDestroyMatDescr.restype = int
_libcusparse.cusparseDestroyMatDescr.argtypes = [ctypes.c_int]
def cusparseDestroyMatDescr(desc):
    """
    Releases the memory allocated for the matrix descriptor.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.

    """

    status = _libcusparse.cusparseDestroyMatDescr(desc)
    cusparseCheckStatus(status)

_libcusparse.cusparseSetMatType.restype = int
_libcusparse.cusparseSetMatType.argtypes = [cusparseMatDescr,
                                            ctypes.c_int]
def cusparseSetMatType(desc, type):
    """
    Sets the matrix type of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.
    type : int
        Matrix type.

    """
    status = _libcusparse.cusparseSetMatType(desc, type)
    cusparseCheckStatus(status)

_libcusparse.cusparseGetMatType.restype = int
_libcusparse.cusparseGetMatType.argtypes = [cusparseMatDescr]                                 
def cusparseGetMatType(desc):
    """
    Gets the matrix type of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.

    Returns
    -------
    type : int
        Matrix type.

    """

    return _libcusparse.cusparseGetMatType(desc)

_libcusparse.cusparseSetMatIndexBase.restype = int
_libcusparse.cusparseSetMatIndexBase.argtypes = [cusparseMatDescr,
                                                 ctypes.c_int]
def cusparseSetMatIndexBase(desc, index_base=CUSPARSE_INDEX_BASE_ZERO):
    """
    Sets the matrix index base of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.
    index_base : int
        Matrix base index.

    """

    if index_base not in [CUSPARSE_INDEX_BASE_ZERO,
                          CUSPARSE_INDEX_BASE_ONE]:
        raise ValueError("invalid index_base.  base index of matrix must be 0"
                         " or 1")
    status = _libcusparse.cusparseSetMatIndexBase(desc, index_base)
    cusparseCheckStatus(status)

_libcusparse.cusparseGetMatIndexBase.restype = int
_libcusparse.cusparseGetMatIndexBase.argtypes = [cusparseMatDescr]                                 
def cusparseGetMatIndexBase(desc):
    """
    Gets the matrix index base of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.

    Returns
    -------
    index_base : int
        Matrix base index.

    """

    return _libcusparse.cusparseGetMatIndexBase(desc)

# cusparseStatus_t 
# cusparseSnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, 
#              int n, const cusparseMatDescr_t descrA, 
#              const float           *A, 
#              int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr)

# Format conversion functions:
_libcusparse.cusparseSnnz.restype = int
_libcusparse.cusparseSnnz.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      cusparseMatDescr,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p]
def cusparseSnnz(handle, dirA, m, n, descrA, A, lda, 
                 nnzPerRowColumn, nnzTotalDevHostPtr):
    """
    Compute number of non-zero elements per row, column, or dense matrix.

    Parameters
    ----------
    handle : int
        CUSPARSE library context.
    dirA : int
        Data direction of elements.
    m : int
        Rows in A.
    n : int
        Columns in A.
    descrA : cusparseMatDescr
        Matrix descriptor.
    A : pycuda.gpuarray.GPUArray
        Dense matrix of dimensions (lda, n).
    lda : int
        Leading dimension of A.
    
    Returns
    -------
    nnzPerRowColumn : pycuda.gpuarray.GPUArray
        Array of length m or n containing the number of 
        non-zero elements per row or column, respectively.
    nnzTotalDevHostPtr : pycuda.gpuarray.GPUArray
        Total number of non-zero elements in device or host memory.

    """

    # Unfinished:
    nnzPerRowColumn = gpuarray.empty()
    nnzTotalDevHostPtr = gpuarray.empty()

    status = _libcusparse.cusparseSnnz(handle, dirA, m, n, 
                                       descrA, int(A), lda,
                                       int(nnzPerRowColumn), int(nnzTotalDevHostPtr))
    cusparseCheckStatus(status)
    return nnzPerVector, nnzHost

_libcusparse.cusparseSdense2csr.restype = int
_libcusparse.cusparseSdense2csr.argtypes = [ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            cusparseMatDescr,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]
def cusparseSdense2csr(handle, m, n, descrA, A, lda, 
                       nnzPerRow, csrValA, csrRowPtrA, csrColIndA):
    # Unfinished
    pass


# cusparseStatus_t 
# cusparseScsrmv(cusparseHandle_t handle, cusparseOperation_t transA, 
#                int m, int n, int nnz, const float           *alpha, 
#                const cusparseMatDescr_t descrA, 
#                const float           *csrValA, 
#                const int *csrRowPtrA, const int *csrColIndA,
#                const float           *x, const float           *beta, 
#                float           *y)
# cusparseStatus_t 
# cusparseDcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, 
#                int m, int n, int nnz, const double          *alpha, 
#                const cusparseMatDescr_t descrA, 
#                const double          *csrValA, 
#                const int *csrRowPtrA, const int *csrColIndA,
#                const double          *x, const double          *beta, 
#                double          *y)
# cusparseStatus_t 
# cusparseCcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, 
#                int m, int n, int nnz, const cuComplex       *alpha, 
#                const cusparseMatDescr_t descrA, 
#                const cuComplex       *csrValA, 
#                const int *csrRowPtrA, const int *csrColIndA,
#                const cuComplex       *x, const cuComplex       *beta, 
#                cuComplex       *y)
# cusparseStatus_t 
# cusparseZcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, 
#                int m, int n, int nnz, const cuDoubleComplex *alpha, 
#                const cusparseMatDescr_t descrA, 
#                const cuDoubleComplex *csrValA, 
#                const int *csrRowPtrA, const int *csrColIndA, 
#                const cuDoubleComplex *x, const cuDoubleComplex *beta, 
#                cuDoubleComplex *y)

# Format conversion functions:
_libcusparse.cusparseCcsrmv.restype = int
_libcusparse.cusparseCcsrmv.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      cusparseMatDescr,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p]
def cusparseCcsrmv(handle, op, m, n, nnz, alpha, descrA, csrValA,
                   csrRowPtrA, csrColIndA, x, beta, y):
    """
    Sparse matrix vector multiplication.

    y = alpha * op(A) ∗ x + beta ∗ y

    Parameters
    ----------
    handle : int
        CUSPARSE library context.
    op : int
        the operation to perform
    m : int
        Rows in A.
    n : int
        Columns in A.
    nnz : int
        Number of non-zero entries in A
    alpha : cuComplex
        Scalar used for multiplication of x
    descrA : cusparseMatDescr
        Matrix descriptor.
    csrValA : pycuda.gpuarray.GPUArray
        CUSPARSE CSR Matrix values
    csrRowPtrA : pycuda.gpuarray.GPUArray
        CUSPARSE CSR Matrix row pointers
    csrColIndA : pycuda.gpuarray.GPUArray
        CUSPARSE CSR Matrix column indices
    x : pycuda.gpuarray.GPUArray
        vector to multiply by A
    beta : cuComplex
        Scalar used for multiplication of y
    y : pycuda.gpuarray.GPUArray
        vector to add
    """

    # Unfinished:
    status = _libcusparse.cusparseCcsrmv(
        handle, op, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
        x, beta, y)
    cusparseCheckStatus(status)
    return
