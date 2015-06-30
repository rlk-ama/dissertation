#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

from libc.math cimport sqrt

cpdef double func_loc_state(double arg, double phi):
    return phi*arg

cpdef double func_scale_state(double sigma):
    return sigma

cpdef double func_scale_obs(double sigma):
    return sigma

cpdef double func_loc_proposal(double[::1] args, double phi, double scale_state, double scale_obs, double scale):
    cdef double output = scale*(phi*args[0]/(scale_state*scale_state) + args[1]/(scale_obs*scale_obs))
    return output

cpdef double func_scale_proposal(double scale):
    return sqrt(scale)