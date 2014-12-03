#!/usr/bin/env python

"""
Python interface to cuSPARSE functions.

Note: You may need to set the environment variable CUDA_ROOT to the base of
your CUDA installation.
"""

# import low level cuSPARSE python wrappers and constants
from ._cusparse_cffi import *
