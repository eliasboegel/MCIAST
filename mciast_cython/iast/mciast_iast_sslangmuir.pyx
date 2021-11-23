cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport fabs, log

cdef f(uarr, mixture):
    cdef double[:] farr = np.empty(mixture.n_components - 1, dtype=np.double) # Result array
    cdef double[:] spreading_pressures = np.empty(mixture.n_components, dtype=np.double)

    cdef int i = 0
    for i in range(uarr.shape[0] - 1):
        spreading_pressures[i] = mixture.isotherm_data[i,0] / mixture.isotherm_data[i,1] * log(fabs(1 + mixture.isotherm_data[i,1] * mixture.partial_pressures[i] / uarr[i])) # Fill with spreading pressures

    #spreading_pressures[:-1] = mixture.isotherm_data[:-1,0] / mixture.isotherm_data[:-1,1] * np.log(np.abs(1 + mixture.isotherm_data[:-1,1] * mixture.partial_pressures[:-1] / uarr)) # Fill with spreading pressures
    spreading_pressures[-1] = mixture.isotherm_data[-1,0] / mixture.isotherm_data[-1,1] * np.log(np.abs(1 + mixture.isotherm_data[-1,1] * mixture.partial_pressures[-1] / (1 - np.sum(uarr)))) # Fill last one using mole fraction = 1 - sum of all other mole fractions

    for i in range(mixture.n_components-1):
        farr[i] = spreading_pressures[i] - spreading_pressures[i+1]

    return farr

cdef loadings(pressures, mixture):
    #cdef double[:] res = np.empty(mixture.n_components, dtype=np.double)

    return mixture.isotherm_data[:,0] * pressures / (1 + mixture.isotherm_data[:,1] * pressures)