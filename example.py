import mciast as mc
import numpy as np

n_components = 2

gas = mc.Mixture(1000*np.ones(n_components), "sslangmuir", np.ones((n_components, 2)))

mc.solve_iast(gas)