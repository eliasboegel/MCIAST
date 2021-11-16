import platform, os
import mciast_iast

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=1000)

class Mixture:
    n_components = None
    __available_models = ['sslangmuir']
    partial_pressures = None
    isotherm_model = None
    isotherm_data = None

    def __init__(self, partial_pressures, isotherm_model, isotherm_data):
        self.n_components = partial_pressures.shape[0]
        self.partial_pressures = partial_pressures
        self.isotherm_model = isotherm_model
        self.isotherm_data = isotherm_data

def solve_iast(Mixture):
    return mciast_iast.__iast_solve(Mixture)
    