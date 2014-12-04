from scikits.cuda.cusparse import *

import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_almost_equal

import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import scipy.sparse


def test_context_create_destroy():
    handle = cusparseCreate()
    cusparseDestroy(handle)


def test_get_version():
    handle = cusparseCreate()
    version = cusparseGetVersion(handle)
    assert type(version) == int
    cusparseDestroy(handle)


def test_get_set_PointerMode():
    handle = cusparseCreate()

    # test default mode
    mode = cusparseGetPointerMode(handle)
    assert mode == CUSPARSE_POINTER_MODE_HOST

    # test setting/getting new mode
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE)
    mode = cusparseGetPointerMode(handle)
    assert mode == CUSPARSE_POINTER_MODE_DEVICE

    # can't set outside enumerated range
    assert_raises(CUSPARSE_STATUS_INVALID_VALUE, cusparseSetPointerMode,
                  handle, 2)

    cusparseDestroy(handle)


def test_matrix_descriptor_create_get_set_destroy():
    #create matrix description
    descrA = cusparseCreateMatDescr()

    #get default values/set
    assert cusparseGetMatType(descrA) == CUSPARSE_MATRIX_TYPE_GENERAL
    assert cusparseGetMatDiagType(descrA) == CUSPARSE_DIAG_TYPE_NON_UNIT
    assert cusparseGetMatIndexBase(descrA) == CUSPARSE_INDEX_BASE_ZERO
    assert cusparseGetMatFillMode(descrA) == CUSPARSE_FILL_MODE_LOWER

    #test set/get new values
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_HERMITIAN)
    assert cusparseGetMatType(descrA) == CUSPARSE_MATRIX_TYPE_HERMITIAN
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_UNIT)
    assert cusparseGetMatDiagType(descrA) == CUSPARSE_DIAG_TYPE_UNIT
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE)
    assert cusparseGetMatIndexBase(descrA) == CUSPARSE_INDEX_BASE_ONE
    cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER)
    assert cusparseGetMatFillMode(descrA) == CUSPARSE_FILL_MODE_UPPER

    # can't set outside enumerated range
    assert_raises(
        OverflowError, cusparseSetMatType,descrA,-1)
    assert_raises(
        CUSPARSE_STATUS_INVALID_VALUE, cusparseSetMatType, descrA, 100)
    assert_raises(
        CUSPARSE_STATUS_INVALID_VALUE, cusparseSetMatDiagType, descrA, 100)
    assert_raises(
        CUSPARSE_STATUS_INVALID_VALUE, cusparseSetMatIndexBase, descrA, 100)
    assert_raises(
        CUSPARSE_STATUS_INVALID_VALUE, cusparseSetMatFillMode, descrA, 100)

    # OLD BEHAVIOR:  float input gets cast to int
    # cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL + 0.5)
    # assert cusparseGetMatType(descrA) == CUSPARSE_MATRIX_TYPE_GENERAL
    assert_raises(TypeError, cusparseSetMatType, descrA,
                  CUSPARSE_MATRIX_TYPE_GENERAL + 0.5)
    #destroy
    cusparseDestroyMatDescr(descrA)


def test_dense_nnz():
    from numpy.testing import assert_almost_equal, assert_equal
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    A = gpuarray.to_gpu(A_cpu)

    handle = cusparseCreate()
    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    try:
        cusparse_dtypes = [np.float32, np.float64, np.complex64, np.complex128]
        for dirA in [CUSPARSE_DIRECTION_ROW, CUSPARSE_DIRECTION_COLUMN]:
            for dtype in cusparse_dtypes:
                nnzRowCol, nnzTotal = dense_nnz(
                    handle, descrA, A.astype(dtype), dirA=dirA)
                assert nnzTotal == 5
                if dirA == CUSPARSE_DIRECTION_ROW:
                    assert_equal(nnzRowCol.get(), [3, 0, 1, 1])
                else:
                    assert_equal(nnzRowCol.get(), [1, 2, 2])
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)


def test_csrmv():
    import scipy.sparse
    cusparse_dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    trans_list = [CUSPARSE_OPERATION_NON_TRANSPOSE,
                  CUSPARSE_OPERATION_TRANSPOSE,
                  CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE]
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    A = gpuarray.to_gpu(A_cpu)

    handle = cusparseCreate()
    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    csr_numpy = scipy.sparse.csr_matrix(A_cpu)
    indptr = csr_numpy.indptr
    indices = csr_numpy.indices
    csr_data = csr_numpy.data

    csrRowPtrA = gpuarray.to_gpu(indptr)
    csrColIndA = gpuarray.to_gpu(indices)
    m, n = csr_numpy.shape
    alpha = 2.0
    try:
        for transA in trans_list:
            for dtype in cusparse_dtypes:
                if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                    x = gpuarray.to_gpu(np.ones((n, ), dtype=dtype))
                else:
                    x = gpuarray.to_gpu(np.ones((m, ), dtype=dtype))
                csrValA = gpuarray.to_gpu(csr_data.astype(dtype))

                # test mutliplication without passing in y
                beta = 0.0
                y = csrmv(handle, descrA, csrValA, csrRowPtrA, csrColIndA, m,
                          n, x, transA=transA, alpha=alpha, beta=beta)
                y_cpu = y.get()

                # repeat, but pass in previous y with beta = 1.0
                beta = 1.0
                y = csrmv(handle, descrA, csrValA, csrRowPtrA, csrColIndA, m,
                          n, x, transA=transA, alpha=alpha, beta=beta, y=y)
                y_cpu2 = y.get()
                if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                    assert_almost_equal(y_cpu, [2., 2., 4., 6.])
                    assert_almost_equal(y_cpu2, 2 * y_cpu)
                else:
                    assert_almost_equal(y_cpu, [4., 2., 8.])
                    assert_almost_equal(y_cpu2, 2 * y_cpu)
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)

def run_all():
    print("Testing context creation")
    test_context_create_destroy()
    print("Testing get version")
    test_get_version()
    print("Testing PointerMode")
    test_get_set_PointerMode()
    print("Testing Matrix Descriptor")
    test_matrix_descriptor_create_get_set_destroy()
    print("Testing dense nnz")
    test_dense_nnz()
    print("Testing CSR mv")
    test_csrmv()