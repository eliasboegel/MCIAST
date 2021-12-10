import mciast as mc
import numpy as np
import pyiast
import pandas as pd
import os, time

dirpath = os.path.abspath(os.path.dirname(__file__))


df_N2 = pd.read_csv(dirpath + "/n2.csv", skiprows=1)
N2_isotherm = pyiast.ModelIsotherm(df_N2, loading_key="Loading(mmol/g)", pressure_key="P(bar)", model="Langmuir")
#pyiast.plot_isotherm(N2_isotherm)
#N2_isotherm.print_params()

df_CO2 = pd.read_csv(dirpath + "/co2.csv", skiprows=1)
CO2_isotherm = pyiast.ModelIsotherm(df_CO2, loading_key="Loading(mmol/g)", pressure_key="P(bar)", model="Langmuir")
#CO2_isotherm.print_params()
#pyiast.plot_isotherm(CO2_isotherm)

isotherms = np.array([
    [N2_isotherm.params['M']*N2_isotherm.params['K'], N2_isotherm.params['K']],
    [CO2_isotherm.params['M']*CO2_isotherm.params['K'], CO2_isotherm.params['K']]
])
partial_pressures = np.array([0.679, 0.166])

starttime = time.time()
for i in range(1000):
    pyiast.iast(partial_pressures, [N2_isotherm, CO2_isotherm], warningoff=True)
print(f"pyIAST: {time.time() - starttime}")

gas = mc.Mixture(partial_pressures, "sslangmuir", isotherms)

starttime = time.time()
for i in range(1000):
    mc.solve_iast(gas, 1e-9)
print(f"MCIAST: {time.time() - starttime}")