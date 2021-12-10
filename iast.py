import scipy as sp
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=1000)

def solve(partial_pressures, isotherm_data):
    # Allocate memory for calculation caches once and pass as parameter to avoid constant allocation/deallocation on every call of __func
    func_cache = np.empty(partial_pressures.shape[0], dtype=np.double)

    #Precompute K*partial_pressures as it is used multiple times
    pre_calc_kpress = isotherm_data[:,1] * partial_pressures
    pre_calc_kmpress = pre_calc_kpress * isotherm_data[:,0]

    # Initial guess is single component loadings normalized
    uarr = pre_calc_kmpress / (1 + pre_calc_kpress)
    uarr = uarr / np.sum(uarr)

    # Solve system
    subsol = sp.optimize.root(__func, uarr[:-1], args=(isotherm_data, func_cache, pre_calc_kpress), method='lm').x

    uarr[:] = subsol
    uarr[-1] = 1 - np.sum(subsol)

    one_over_sc_loadings = (uarr + pre_calc_kpress) / pre_calc_kmpress
    tot_loading = 1 / np.dot(uarr, one_over_sc_loadings) # 1 / sum(uarr / sc_loadings)

    return tot_loading * uarr # Multicomponent loadings in equilibrium conditions in moles

def __func(uarr, isotherm_data, calc_cache, pre_calcs):
    # Create vector of all adsorbed mole fractions
    calc_cache[:-1] = uarr
    calc_cache[-1] = 1 - np.sum(uarr)

    # Overwrite vector with all spreading pressure terms
    calc_cache = isotherm_data[:,0] * np.log1p(pre_calcs / calc_cache) # Removed abs and seems to work as numbers here seem to always be positive

    return calc_cache[1:] - calc_cache[:-1] # Return differences between spreading pressures in staggered way