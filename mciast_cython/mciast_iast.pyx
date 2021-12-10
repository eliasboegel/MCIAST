import numpy as np

def __newton_jacobian_sslanguir(uarr, lang_coeffs):
    n = uarr.shape[0] // 2 # Number of components

    # Result array
    jarr = np.zeros((uarr.shape[0], uarr.shape[0]), dtype=uarr.dtype)
    

    # Fill top left diag and right -1 column
    for i in range(n):
        jarr[i, i] = -uarr[n + i]
        jarr[i, 2*n-1] = -1

    # Fill bottom right diags
    cached_result = 0
    jarr[n,n] = lang_coeffs[0,0]/lang_coeffs[0,1] * 1/(1 + lang_coeffs[0,1] * uarr[n]) # Single element in n-th row
    for i in range(1, n-1):
        jarr[n+i, n+i] = lang_coeffs[i,0]/lang_coeffs[i,1] * 1/(1 + lang_coeffs[i,1] * uarr[n+i])
        jarr[n+i, n+i-1] = - jarr[n+i, n+i]
    jarr[2*n-1,2*n-2] = - lang_coeffs[n-1,0]/lang_coeffs[n-1,1] * 1/(1 + lang_coeffs[n-1,1] * uarr[2*n-1]) # Single element in 2n-th row

    return jarr

def __newton_f_sslangmuir(partial_pressures, uarr, lang_coeffs):
    n = uarr.shape[0] // 2 # Number of components

    # Result array
    farr = np.zeros((uarr.shape[0]), dtype=uarr.dtype)

    # First n elements
    farr[:n] = partial_pressures - uarr[:n] * uarr[n:]

    # n-th to n-1-th elements
    log_terms = np.log(np.abs(1 + lang_coeffs[:,1] * uarr[n:]))
    for i in range(n-1):
        farr[n+i] = lang_coeffs[i,0] / lang_coeffs[i,1] * log_terms[i] - lang_coeffs[i+1,0] / lang_coeffs[i+1,1] * log_terms[i+1]

    # Last element
    farr[2*n-1] = 1 - np.sum(uarr[:n-1])
    
    return farr

def __loadings_sslangmuir(uarr, lang_coeffs):
    n = uarr.shape[0] // 2 # Number of components

    # Result array
    larr = np.zeros(n, dtype=uarr.dtype)

    # Array of single component loadings in equilibrium conditions
    sc_loadings = np.zeros(n, dtype=uarr.dtype)
    for i in range(n):
        sc_loadings[i] = lang_coeffs[i,0] * uarr[n+i] / (1 + lang_coeffs[i,1] * uarr[n+i]) # Single-site langmuir isotherm model

    tot_loading = 1 / np.sum(uarr[:n-1] / sc_loadings)

    larr = tot_loading * uarr[:n-1]
    return larr

def __iast_solve(mixture):
    if (mixture.isotherm_model == 0): 
        print("No Isotherm model")
        return


    uarr = np.ones(2*mixture.isotherm_data.shape[0]) / mixture.n_components # Initial guess
    loading = np.empty(mixture.n_components)

    if (mixture.isotherm_model == "sslangmuir"): # Single-site langmuir isotherm
        for i in range(1): # Fixed number of newton iterations for now
            jac = __newton_jacobian_sslanguir(uarr, mixture.isotherm_data)
            f = __newton_f_sslangmuir(mixture.partial_pressures, uarr, mixture.isotherm_data)

            print(jac)

            uarr = uarr - np.linalg.inv(jac).dot(f)

        loading = __loadings_sslangmuir(uarr, mixture.isotherm_data)

    return loading