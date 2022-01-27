import scipy as sp
import numpy as np
import pyiast
np.set_printoptions(edgeitems=30, linewidth=1000)

def fit(data_list, skipheader=0, skipfooter=0):
    isotherm_data = np.empty((len(data_list),2), dtype=np.double)
    component_names = []

    def n(p, m, k): return m * k * p / (1 + k * p)

    for i in range(len(data_list)):
        data = np.genfromtxt(data_list[i], skip_header=2, delimiter=',')
        with open(data_list[i]) as f:
            component_names.append(f.readline())

        initial_guess = [
            1.1 * np.max(data[:,0]),
            0.1
        ]

        vals = sp.optimize.curve_fit(n, data[:,0], data[:,1], p0=initial_guess, maxfev=1000, method='lm', xtol=1e-15, col_deriv=True)
        isotherm_data[i] = vals[0]

    component_names.append("He")
    return np.array(component_names), isotherm_data

def solve(partial_pressures, params):
    # PYIAST VERSION
    """loadings = np.empty(partial_pressures.shape[0], dtype=np.double)

    for i in range(partial_pressures.shape[0]):
        loadings[i] = params.isotherms[i].loading(partial_pressures[i])

    return loadings"""

    # MCIAST VERSION
    
    #Precompute K*partial_pressures as it is used multiple times
    pre_calc_kpress = params.isotherms[:,1] * partial_pressures
    pre_calc_kmpress = pre_calc_kpress * params.isotherms[:,0]

    # Initial guess is single component loadings normalized
    #uarr = pre_calc_kmpress / (1 + pre_calc_kpress)
    uarr = params.isotherms[:,0] * params.isotherms[:,1] * partial_pressures / (1 + params.isotherms[:,1] * partial_pressures)
    uarr = uarr / np.sum(uarr)

    # Solve system
    subsol = sp.optimize.root(__func, uarr[:-1], args=(params, pre_calc_kpress), method='lm', options={'xtol': 1e-15, 'col_deriv': True}).x

    uarr[:] = subsol
    uarr[-1] = 1 - np.sum(subsol)

    #one_over_sc_loadings = (uarr + pre_calc_kpress) / pre_calc_kmpress
    #tot_loading = 1 / np.dot(uarr, one_over_sc_loadings) # 1 / sum(uarr / sc_loadings)

    #sc_loadings = pre_calc_kmpress / (1 + pre_calc_kpress)
    sc_loadings = params.isotherms[:,0] * params.isotherms[:,1] * partial_pressures / (1 + params.isotherms[:,1] * partial_pressures)

    inv_loading = np.sum(uarr / sc_loadings)

    tot_loading = 1 / inv_loading

    return tot_loading * uarr # Multicomponent loadings in equilibrium conditions in moles

def __func(uarr, params, pre_calcs):
    # Create vector of all adsorbed mole fractions
    params.calc_cache[:-1] = uarr
    params.calc_cache[-1] = 1 - np.sum(uarr)

    # Overwrite vector with all spreading pressure terms
    params.calc_cache = params.isotherms[:,0] * np.log1p(np.abs(pre_calcs / params.calc_cache))

    return params.calc_cache[1:] - params.calc_cache[:-1] # Return differences between spreading pressures in staggered way
