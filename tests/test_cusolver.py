from __future__ import division

from scikits.cuda.cusolver import *

from scikits.cuda import cusolver
cusolver.init()

import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_almost_equal

from unittest import skipIf

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

cusparse_real_dtypes = [np.float32, np.float64]
cusparse_complex_dtypes = [np.complex64, np.complex128]
cusparse_dtypes = cusparse_real_dtypes + cusparse_complex_dtypes



# def test_context_create_destroy():
#     handle = cusparseCreate()
#     cusparseDestroy(handle)
