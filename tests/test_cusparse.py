from scikits.cuda.cusparse import *
from scikits.cuda.cusparse import _csrgemmNnz

import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_almost_equal

import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import scipy.sparse  # TODO: refactor to remove this

cusparse_dtypes = [np.float32, np.float64, np.complex64, np.complex128]
trans_list = [CUSPARSE_OPERATION_NON_TRANSPOSE,
              CUSPARSE_OPERATION_TRANSPOSE,
              CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE]


def test_context_create_destroy():
    handle = cusparseCreate()
    cusparseDestroy(handle)


def test_get_version():
    handle = cusparseCreate()
    try:
        version = cusparseGetVersion(handle)
        assert type(version) == int
    finally:
        cusparseDestroy(handle)


def test_create_destroy_hyb():
    HybA = cusparseCreateHybMat()
    cusparseDestroyHybMat(HybA)


def test_get_set_PointerMode():
    handle = cusparseCreate()
    try:
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
    finally:
        cusparseDestroy(handle)


def test_matrix_descriptor_create_get_set_destroy():
    #create matrix description
    descrA = cusparseCreateMatDescr()

    try:
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
    finally:
        cusparseDestroyMatDescr(descrA)


def test_dense_nnz():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    A = gpuarray.to_gpu(A_cpu)

    handle = cusparseCreate()
    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    #loop over all directions and dtypes
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


def test_dense2csr_csr2dense():
    m = 100
    A = np.eye(m)
    for dtype in cusparse_dtypes:
        A = A.astype(dtype)
        (handle, descrA, csrValA, csrRowPtrA, csrColIndA) = dense2csr(A)
        try:
            assert_equal(csrValA.get(), np.ones((m,)))
            assert_equal(csrRowPtrA.get(), np.arange(m+1))
            assert_equal(csrColIndA.get(), np.arange(m))

            A_dense = csr2dense(handle, m, m, descrA, csrValA, csrRowPtrA,
                                csrColIndA)
            assert_equal(A, A_dense.get())
            # release handle, descrA that were generated within dense2csr
        finally:
            cusparseDestroy(handle)
            cusparseDestroyMatDescr(descrA)



def test_csrmv():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    # A = gpuarray.to_gpu(A_cpu)

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
    #loop over all transpose operations and dtypes
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


def test_csrmm():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])

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
    n = 5
    alpha = 2.0

    #loop over all transpose operations and dtypes
    try:
        for transA in trans_list:
            for dtype in cusparse_dtypes:
                csrValA = gpuarray.to_gpu(csr_data.astype(dtype))

                m, k = A_cpu.shape
                if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                    m, k = A_cpu.shape
                    B_cpu = np.ones((k, n), dtype=dtype, order='F')
                    expected_result = alpha * np.dot(A_cpu, B_cpu)
                else:
                    B_cpu = np.ones((m, n), dtype=dtype, order='F')
                    if transA == CUSPARSE_OPERATION_TRANSPOSE:
                        expected_result = alpha * np.dot(A_cpu.T, B_cpu)
                    else:
                        expected_result = alpha * np.dot(np.conj(A_cpu).T,
                                                         B_cpu)
                B = gpuarray.to_gpu(B_cpu)
                # test mutliplication without passing in C
                beta = 0.0
                C = csrmm(handle, m, n, k, descrA, csrValA, csrRowPtrA,
                          csrColIndA, B, transA=transA, alpha=alpha,
                          beta=beta)
                C_cpu = C.get()
                assert_almost_equal(C_cpu, expected_result)

                # repeat, but pass in previous y with beta = 1.0
                beta = 1.0
                C = csrmm(handle, m, n, k, descrA, csrValA, csrRowPtrA,
                          csrColIndA, B, C=C, transA=transA, alpha=alpha,
                          beta=beta)
                C_cpu2 = C.get()
                assert_almost_equal(C_cpu2, 2*expected_result)
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)


def test_csrmm2():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])

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
    n = 5
    alpha = 2.0
    try:
        for transB in trans_list[:-1]:
            for transA in trans_list:
                for dtype in cusparse_dtypes:
                    csrValA = gpuarray.to_gpu(csr_data.astype(dtype))

                    m, k = A_cpu.shape
                    if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                        m, k = A_cpu.shape

                        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
                            B_cpu = np.ones((k, n), dtype=dtype, order='F')
                            opB = B_cpu
                        else:
                            B_cpu = np.ones((n, k), dtype=dtype, order='F')
                            opB = B_cpu.T
                        opA = A_cpu
                    else:
                        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
                            B_cpu = np.ones((m, n), dtype=dtype, order='F')
                            opB = B_cpu
                        else:
                            # cuSPARSE doesn't implement this case
                            continue
                            # B_cpu = np.ones((n, m), dtype=dtype, order='F')
                            # opB = B_cpu.T
                        if transA == CUSPARSE_OPERATION_TRANSPOSE:
                            opA = A_cpu.T
                        else:
                            opA = np.conj(A_cpu).T

                    expected_result = alpha * np.dot(opA, opB)
                    B = gpuarray.to_gpu(B_cpu)

                    # test mutliplication without passing in C
                    beta = 0.0
                    C = csrmm2(handle, m, n, k, descrA, csrValA, csrRowPtrA,
                               csrColIndA, B, transA=transA, transB=transB,
                               alpha=alpha, beta=beta)
                    C_cpu = C.get()
                    assert_almost_equal(C_cpu, expected_result)

                    # repeat, but pass in previous y with beta = 1.0
                    beta = 1.0
                    C = csrmm2(handle, m, n, k, descrA, csrValA, csrRowPtrA,
                               csrColIndA, B, C=C, transA=transA,
                               transB=transB, alpha=alpha, beta=beta)
                    C_cpu2 = C.get()
                    assert_almost_equal(C_cpu2, 2*expected_result)
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)


def test_csrmmNnz():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    # A = gpuarray.to_gpu(A_cpu)

    handle = cusparseCreate()
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

    descrB = cusparseCreateMatDescr()
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO)

    descrC = cusparseCreateMatDescr()
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL)

    csr_numpy = scipy.sparse.csr_matrix(A_cpu)
    indptr = csr_numpy.indptr
    indices = csr_numpy.indices
    csr_data = csr_numpy.data

    B_cpu = csr_numpy.T.tocsr()

    csrRowPtrA = gpuarray.to_gpu(indptr)
    csrColIndA = gpuarray.to_gpu(indices)

    csrRowPtrB = gpuarray.to_gpu(B_cpu.indptr)
    csrColIndB = gpuarray.to_gpu(B_cpu.indices)

    m, k = A_cpu.shape

    #loop over all transpose operations and dtypes
    try:
        for transA in trans_list:
            transB = transA

            if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                m, k = A_cpu.shape
            else:
                k, m = A_cpu.shape

            if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
                kB, n = B_cpu.shape
            else:
                n, kB = B_cpu.shape

            csrRowPtrC = gpuarray.to_gpu(np.zeros((m+1, ), dtype=np.int32))
            nnzC = _csrgemmNnz(handle, m, n, k, descrA, csrRowPtrA, csrColIndA,
                              descrB, csrRowPtrB, csrColIndB, descrC,
                              csrRowPtrC, nnzA=None, nnzB=None,
                              transA=transA,
                              transB=transB,
                              check_inputs=True)
            if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                assert nnzC == 8
                assert_equal(csrRowPtrC.get(), [0, 2, 3, 6, 8])
            else:
                assert nnzC == 5
                assert_equal(csrRowPtrC.get(), [0, 2, 3, 5])
    finally:
        cusparseDestroy(handle)
        cusparseDestroyMatDescr(descrA)
        cusparseDestroyMatDescr(descrB)
        cusparseDestroyMatDescr(descrC)


def test_csrgemm():
    A_cpu = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 3]])
    # A = gpuarray.to_gpu(A_cpu)

    handle = cusparseCreate()
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST)

    descrA = cusparseCreateMatDescr()
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

    descrB = cusparseCreateMatDescr()
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO)

    descrC = cusparseCreateMatDescr()
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL)

    csr_numpy = scipy.sparse.csr_matrix(A_cpu)
    indptr = csr_numpy.indptr
    indices = csr_numpy.indices
    csr_data = csr_numpy.data

    B_cpu = csr_numpy.T.tocsr()

    csrRowPtrA = gpuarray.to_gpu(indptr)
    csrColIndA = gpuarray.to_gpu(indices)

    csrRowPtrB = gpuarray.to_gpu(B_cpu.indptr)
    csrColIndB = gpuarray.to_gpu(B_cpu.indices)

    m, k = A_cpu.shape

    for transA in trans_list:
        transB = transA

        if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
            m, k = A_cpu.shape
        else:
            k, m = A_cpu.shape
        if transB == CUSPARSE_OPERATION_NON_TRANSPOSE:
            kB, n = B_cpu.shape
        else:
            n, kB = B_cpu.shape

        for dtype in cusparse_dtypes:
            csrValA = gpuarray.to_gpu(csr_data.astype(dtype))
            csrValB = gpuarray.to_gpu(B_cpu.data.astype(dtype))

            descrC, csrValC, csrRowPtrC, csrColIndC = csrgemm(
                handle, m, n, k, descrA, csrValA, csrRowPtrA, csrColIndA,
                descrB, csrValB, csrRowPtrB, csrColIndB, transA=transA,
                transB=transB)
            if transA == CUSPARSE_OPERATION_NON_TRANSPOSE:
                assert_almost_equal(csrValC.get(), [1, 1, 1, 1, 2, 3, 3, 9])
            else:
                assert_almost_equal(csrValC.get(), [2, 1, 1, 1, 10])


def run_all():
    print("Testing context creation")
    test_context_create_destroy()
    print("Test HYB matrix creation")
    test_create_destroy_hyb()
    print("Testing get version")
    test_get_version()
    print("Testing PointerMode")
    test_get_set_PointerMode()
    print("Testing Matrix Descriptor")
    test_matrix_descriptor_create_get_set_destroy()
    print("Testing dense nnz")
    test_dense_nnz()
    print("Testing dense2csr and csr2dense")
    test_dense2csr_csr2dense()
    print("Testing CSR mv")
    test_csrmv()
    print("Testing CSR mm")
    test_csrmm()
    print("Testing CSR mm2")
    test_csrmm2()
    print("Testing CSR gemmNnz")
    test_csrgemmNnz()
    print("Testing CSR gemm")
    test_csrgemm()
