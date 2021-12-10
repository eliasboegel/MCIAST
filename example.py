import iast
import numpy as np
import pyiast
import pandas as pd
import os

import pstats, cProfile



dirpath = os.path.abspath(os.path.dirname(__file__))


df_N2 = pd.read_csv(dirpath + "/n2.csv", skiprows=1)
N2_isotherm = pyiast.ModelIsotherm(df_N2, loading_key="Loading(mmol/g)", pressure_key="P(bar)", model="Langmuir")
N2_isotherm.print_params()

df_CO2 = pd.read_csv(dirpath + "/co2.csv", skiprows=1)
CO2_isotherm = pyiast.ModelIsotherm(df_CO2, loading_key="Loading(mmol/g)", pressure_key="P(bar)", model="Langmuir")
CO2_isotherm.print_params()


isotherms = np.array([
    [N2_isotherm.params['M'], N2_isotherm.params['K']],
    [CO2_isotherm.params['M'], CO2_isotherm.params['K']]
])
partial_pressures = np.array([0.679, 0.166])


cProfile.runctx("for i in range(1000): pyiast.iast(partial_pressures, [N2_isotherm, CO2_isotherm], warningoff=True)", globals(), locals(), "pyiast.prof")
s = pstats.Stats("pyiast.prof")
s.strip_dirs().sort_stats("time").print_stats()

cProfile.runctx("for i in range(1000): iast.solve(partial_pressures, isotherms)", globals(), locals(), "mciast.prof")
s = pstats.Stats("mciast.prof")
s.strip_dirs().sort_stats("time").print_stats()

print(pyiast.iast(partial_pressures, [N2_isotherm, CO2_isotherm], warningoff=True))
print(iast.solve(partial_pressures, isotherms))