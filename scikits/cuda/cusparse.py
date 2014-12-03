#!/usr/bin/env python

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
             "your CUDA installation")
    raise ImportError(estr)
