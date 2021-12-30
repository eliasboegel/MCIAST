import unittest

import numpy as np

from src.linearized_system import LinearizedSystem
from src.solver import Solver
from src.system_parameters import SysParams


class TestMMS(unittest.TestCase):
    params = SysParams()
    params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.2, 0.8]), n_points=11, p_in=5.0, p_out=5.0, temp=313,
                       c_len=4, u_in=2, void_frac=0.6, disp=[16, 8], kl=[6, 8], rho_p=2, dispersion_helium=8)

    def test_pt_calculation(self):
        ...

    def test_pi_calculation(self):
        ...

    def test_dpi_dt_calculation(self):
        ...

    def test_dpi_dx_calculation(self):
        ...

    def test_d2pi_dx2_calculation(self):
        ...

    def test_v_calculation(self):
        ...

    def test_dv_dx_calculation(self):
        ...

    def test_dpv_dx_calculation(self):
        ...

    def test_S_nu_calculation(self):
        ...

    def test_S_pi_calculation(self):
        ...

