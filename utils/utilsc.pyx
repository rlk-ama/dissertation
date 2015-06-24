#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

import numpy as np
cimport numpy as np

cpdef double[::1] wrapper_arr(func, double[::1] args):
    cdef int dim = args.shape[0]
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        output[i] = func(args[i])
    return output

cpdef double[::1] wrapper_arr_arr(func, double[:, ::1] args):
    cdef int dim = args.shape[0]
    cdef double[::1] output = np.empty(dim)
    cdef int i
    for i in range(dim):
        output[i] = func(args[i])
    return output