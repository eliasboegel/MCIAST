import unittest

import numpy as np

from src.linearized_system import LinearizedSystem
from src.solver import Solver
from src.system_parameters import SysParams
from src.mms import MMS


class TestMMS(unittest.TestCase):
    def test_pt_calculation(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.25, 0.25, 0.25, 0.25]), n_points=5, p_in=1.0,
                           p_out=0.5,
                           temp=288, c_len=1, u_in=1, void_frac=0.1, disp=[0.001, 0.001, 0.001, 0.001], kl=[1, 1, 1, 1],
                           rho_p=1000, mms=True, ms_pt_distribution="linear", mms_mode="transient",
                           mms_convergence_factor=1000)
        mms = MMS(params)
        mms.update_source_functions(100)

        np.testing.assert_array_equal(params.p_total, np.array([0.875, 0.75, 0.625, 0.5]))

    def test_pi_calculation(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.25, 0.25, 0.25, 0.25]), n_points=5, p_in=1.0,
                           p_out=0.5,
                           temp=288, c_len=1, u_in=1, void_frac=0.1, disp=[0.001, 0.001, 0.001, 0.001], kl=[1, 1, 1, 1],
                           rho_p=1000, mms=True, ms_pt_distribution="linear", mms_mode="transient",
                           mms_convergence_factor=1000)
        mms = MMS(params)
        mms.update_source_functions(100)

        np.testing.assert_allclose(mms.pi_matrix, np.array([[1.241250107, -0.8037501065, 1.241250107, -0.8037501065],
                                                            [1.799444199, -1.424444199, 1.799444199, -1.424444199],
                                                            [1.719946207, -1.407446207, 1.719946207, -1.407446207],
                                                            [1.125, -0.875, 1.125, -0.875]]), atol=1e-9)

    def test_dpi_dt_calculation(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.25, 0.25, 0.25, 0.25]), n_points=5, p_in=1.0,
                           p_out=0.5,
                           temp=288, c_len=1, u_in=1, void_frac=0.1, disp=[0.001, 0.001, 0.001, 0.001], kl=[1, 1, 1, 1],
                           rho_p=1000, mms=True, ms_pt_distribution="linear", mms_mode="transient",
                           mms_convergence_factor=1000)
        mms = MMS(params)
        mms.update_source_functions(100)

        np.testing.assert_allclose(mms.dpi_dt_matrix, 1e-6 *
                                   np.array([[-639.8166742, 639.8166742, -639.8166742, 639.8166742],
                                            [-904.837418, 904.837418, -904.837418, 904.837418],
                                            [-639.8166742, 639.8166742, -639.8166742, 639.8166742],
                                            [0.0, 0.0, 0.0, 0.0]]), atol=1e-9)

    def test_dpi_dz_calculation(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.25, 0.25, 0.25]), n_points=5, p_in=1.0,
                           p_out=0.5, y_fill_gas=0.25, disp_fill_gas=0.001, kl_fill_gas=1, temp=288, c_len=1, u_in=1,
                           void_frac=0.1, disp=[0.001, 0.001, 0.001], kl=[1, 1, 1], rho_p=1000, mms=True,
                           ms_pt_distribution="linear", mms_mode="transient", mms_convergence_factor=1000)
        mms = MMS(params)
        mms.update_source_functions(100)

        np.testing.assert_allclose(mms.dpi_dz_matrix,
                                   np.array([[3.336269939, -3.586269939, 3.336269939, -3.586269939],
                                            [0.9857207345, -1.235720735, 0.9857207345, -1.235720735],
                                            [-1.533925633, 1.283925633, -1.533925633, 1.283925633],
                                            [-2.967630585, 2.717630585, -2.967630585, 2.717630585]]), atol=1e-9)

    def test_d2pi_dz2_calculation(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.25, 0.25, 0.25]), n_points=5, p_in=1.0,
                           p_out=0.5, y_fill_gas=0.25, disp_fill_gas=0.001, kl_fill_gas=1, temp=288, c_len=1, u_in=1,
                           void_frac=0.1, disp=[0.001, 0.001, 0.001], kl=[1, 1, 1], rho_p=1000, mms=True,
                           ms_pt_distribution="linear", mms_mode="transient", mms_convergence_factor=1000)
        mms = MMS(params)
        mms.update_source_functions(100)

        np.testing.assert_allclose(mms.d2pi_dz2_matrix,
                                   np.array([[-7.258970985, 7.258970985, -7.258970985, 7.258970985],
                                             [-10.67510341, 10.67510341, -10.67510341, 10.67510341],
                                             [-8.594318838, 8.594318838, -8.594318838, 8.594318838],
                                             [-2.4674011, 2.4674011, -2.4674011, 2.4674011]]), atol=1e-9)

    def test_nu_calculation(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.25, 0.25, 0.25]), n_points=5, p_in=1.0,
                           p_out=0.5, y_fill_gas=0.25, disp_fill_gas=0.001, kl_fill_gas=1, temp=288, c_len=1, u_in=1,
                           void_frac=0.1, disp=[0.001, 0.001, 0.001], kl=[1, 1, 1], rho_p=1000, mms=True,
                           ms_pt_distribution="linear", mms_mode="transient", mms_convergence_factor=1000)
        mms = MMS(params)
        mms.update_source_functions(100)

        np.testing.assert_allclose(mms.nu, np.array([1.261076993, 1.551284027, 0.9904789428, 0.5]), atol=1e-9)

    def test_dnu_dz_calculation(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.25, 0.25, 0.25]), n_points=5, p_in=1.0,
                           p_out=0.5, y_fill_gas=0.25, disp_fill_gas=0.001, kl_fill_gas=1, temp=288, c_len=1, u_in=1,
                           void_frac=0.1, disp=[0.001, 0.001, 0.001], kl=[1, 1, 1], rho_p=1000, mms=True,
                           ms_pt_distribution="linear", mms_mode="transient", mms_convergence_factor=1000)
        mms = MMS(params)
        mms.update_source_functions(100)

        np.testing.assert_allclose(mms.dnu_dz, np.array([2.117017297, -0.5553603673, -3.14318945, 0.0]), atol=1e-9)

    def test_dnupi_dz_calculation(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.25, 0.25, 0.25]), n_points=5, p_in=1.0,
                           p_out=0.5, y_fill_gas=0.25, disp_fill_gas=0.001, kl_fill_gas=1, temp=288, c_len=1, u_in=1,
                           void_frac=0.1, disp=[0.001, 0.001, 0.001], kl=[1, 1, 1], rho_p=1000, mms=True,
                           ms_pt_distribution="linear", mms_mode="transient", mms_convergence_factor=1000)
        mms = MMS(params)
        mms.update_source_functions(100)

        np.testing.assert_allclose(mms.dnupi_dz_matrix,
                                   np.array([[6.835041208, -6.224115387, 6.835041208, -6.224115387],
                                             [0.5297928392, -1.125873984, 0.5297928392, -1.125873984],
                                             [-6.925437812, 5.695571373, -6.925437812, 5.695571373],
                                             [-1.483815283, 1.358815293, -1.483815283, 1.358815293]]), atol=1e-9)

    def test_S_nu_calculation(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.25, 0.25, 0.25]), n_points=5, p_in=1.0,
                           p_out=0.5, y_fill_gas=0.25, disp_fill_gas=0.001, kl_fill_gas=1, temp=288, c_len=1, u_in=1,
                           void_frac=0.1, disp=[0.001, 0.001, 0.001], kl=[1, 1, 1], rho_p=1000, mms=True,
                           ms_pt_distribution="linear", mms_mode="transient", mms_convergence_factor=1000)
        mms = MMS(params)
        mms.update_source_functions(100)

        np.testing.assert_allclose(mms.S_nu,
                                   1e6 * np.array([17.54923263, 35.09846091, 52.64768978, 70.19692214]), atol=1e-9)

    def test_S_pi_calculation(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.25, 0.25, 0.25]), n_points=5, p_in=1.0,
                           p_out=0.5, y_fill_gas=0.25, disp_fill_gas=0.001, kl_fill_gas=1, temp=288, c_len=1, u_in=1,
                           void_frac=0.1, disp=[0.001, 0.001, 0.001], kl=[1, 1, 1], rho_p=1000, mms=True,
                           ms_pt_distribution="linear", mms_mode="transient", mms_convergence_factor=1000)
        mms = MMS(params)
        mms.update_source_functions(100)

        np.testing.assert_allclose(mms.S_pi,
                                   1e6 * np.array([[4.387311472, 4.387304663, 4.387311472, 4.387304663],
                                                  [8.774614054, 8.774616399, 8.774614054, 8.774616399],
                                                  [13.16191742, 13.16192747, 13.16191742, 13.16192747],
                                                  [17.54922912, 17.54923195, 17.54922912, 17.54923195]]), atol=1e-9)

