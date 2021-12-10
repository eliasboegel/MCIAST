import platform, os
import mciast_iast

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=1000)

__available_models = [
    ['None', 0],
    ['sslangmuir', 1]
]

class Mixture:
    n_components = 0
    partial_pressures = np.empty(0)
    isotherm_model = 0
    isotherm_data = np.empty(0)

    def __init__(self, partial_pressures, isotherm_model, isotherm_data):
        self.n_components = partial_pressures.shape[0]
        self.partial_pressures = partial_pressures
        self.isotherm_model = isotherm_model
        self.isotherm_data = isotherm_data
        

def solve_iast(Mixture, max_error):
    return mciast_iast.__iast_solve(Mixture, max_error)
    