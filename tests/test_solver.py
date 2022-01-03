import unittest

import numpy as np

from src.linearized_system import LinearizedSystem
from src.solver import Solver
from src.system_parameters import SysParams


class TestSolver(unittest.TestCase):

    def initialize_solver(self, n_points=10):
        params = SysParams()
        params.init_params(t_end=10000, dt=0.001, y_in=np.asarray([0.5, 0.5]), n_points=n_points, p_in=2e5, temp=298,
                           c_len=1, u_in=1, void_frac=0.995, disp=[0.004, 0.004], kl=[4.35, 1.47], rho_p=1000,
                           p_out=2e5, time_stepping="BE", dimensionless=True, disp_helium=0.004)
        solver = Solver(params)
        return solver

    # def test_initialize_params_correct(self):
    #     params = SysParams()
    #     params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.2, 0.8]), n_points=11, p_in=5.0, p_out=5.0, temp=313,
    #                        c_len=4, u_in=2, void_frac=0.6, disp=[16, 8], kl=[6, 8], rho_p=2, disp_helium=8)
    #     self.assertEqual(params.t_end, 8)
    #     self.assertEqual(params.dt, 0.0005)
    #     self.assertEqual(params.nt, 8000)
    #     self.assertEqual(params.p_in, 5.0)
    #     self.assertEqual(params.p_out, 5.0)
    #     np.testing.assert_array_equal(params.p_total, [5.0] * 10)
    #     self.assertEqual(params.n_points, 11)
    #     np.testing.assert_array_equal(params.y_in, np.asarray([0.2, 0.8, 0.0]))
    #     self.assertEqual(params.temp, 313)
    #     self.assertEqual(params.void_frac, 0.6)
    #     self.assertEqual(params.rho_p, 2)
    #     np.testing.assert_array_equal(params.kl, np.asarray([12, 16, 0]))
    #     np.testing.assert_array_equal(params.disp, np.asarray([2, 1, 1]))
    #     self.assertEqual(params.c_len, 1)
    #     self.assertEqual(params.dz, 0.1)
    #     self.assertEqual(params.dp_dz, 0)
    #     self.assertEqual(params.v_in, 1)
    #     self.assertEqual(params.n_components, 3)

    def test_initialize_solver_matrices(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.2, 0.8]), n_points=4, p_in=5.0, temp=313,
                           c_len=1, u_in=1, void_frac=0.6, disp=[1, 1], kl=[1, 1], rho_p=2,
                           dimensionless=True, p_out=5.0, disp_helium=1)
        solver = Solver(params)
        print("Matrix G: ", solver.params.g_matrix)
        print("Matrix L: ", solver.params.l_matrix)
        print("Matrix D: ", solver.params.d_matrix)
        print("Vector B: ", solver.params.b_v_vector)
        np.testing.assert_allclose(solver.params.g_matrix.toarray(), np.asarray([[0., 1.5, 0.],
                                                                                 [-1.5, 0., 1.5],
                                                                                 [1.5, -6., 4.5]], dtype=float), 1e-5)
        np.testing.assert_allclose(solver.params.l_matrix.toarray(), np.asarray([[-18., 9., 0.],
                                                                                 [9., -18., 9.],
                                                                                 [0., 18., -18.]], dtype=float), 1e-5)
        np.testing.assert_allclose(solver.params.d_matrix, np.asarray([[10.5, 42.0, 0.],
                                                                       [0., 0., 0.],
                                                                       [0., 0., 0.]], dtype=float), 1e-5)
        np.testing.assert_allclose(solver.params.b_v_vector, [-1.5, 0., 0.])
        np.testing.assert_allclose(solver.params.kl, np.asarray([1, 1, 0]))
        np.testing.assert_allclose(solver.params.disp, np.asarray([1, 1, 1]))
        np.testing.assert_allclose(solver.params.y_in, np.asarray([0.2, 0.8, 0]))

    def test_caclulate_dp_dt1(self):
        """
        Case in which kl and disp are identical for each component and equal to 1.
        """
        solver = self.initialize_solver(n_points=4)
        velocities = np.asarray([1, 1, 1])
        p_partial = np.asarray([[1e5, 1e5, 1e5], [0.5e5, 1.5e5, 1e5], [0.6e5, 1.4e5, 1e5]])
        q_eq = np.asarray([[1e-3, 1e-4, 0], [1e-3, 5e-4, 0], [5e-4, 6e-4, 0]])
        q_ads = np.asarray([[1e-4, 1e-5, 0], [1e-4, 1e-3, 0], [1e-4, 1e-4, 0]])
        # print("Matrix G: ", solver.params.g_matrix.toarray())
        # print("Matrix L: ", solver.params.l_matrix.toarray())
        # print("Matrix D: ", solver.params.d_matrix)
        # print("Vector B: ", solver.params.b_v_vector)
        # print("Advection term= :", -solver.params.g_matrix.dot(p_partial))
        # print("Dispersion_term= ",
        #       np.multiply(solver.params.l_matrix.dot(p_partial), np.asarray([0.004, 0.004, 0.004])))
        # print("Adsorpotion term= ",
        #       -8.314 * 298 * ((1 - 0.995) / 0.995) * 1000 * np.multiply(np.asarray([4.35, 1.47, 0.]),
        #                                                                 q_eq - q_ads) + solver.params.d_matrix)
        # print("Final= ",
        #       -solver.params.g_matrix.dot(p_partial) + np.multiply(solver.params.l_matrix.dot(p_partial),
        #                                                            np.asarray([0.004, 0.004, 0.004])) - 8.314 * 298 * (
        #               (1 - 0.995) / 0.995) * 1000 * np.multiply(np.asarray([4.35, 1.47, 0.]),
        #                                                         q_eq - q_ads) + solver.params.d_matrix)
        dp_dt = solver.calculate_dp_dt(velocities,
                                       p_partial,
                                       q_eq,
                                       q_ads)
        np.testing.assert_allclose(dp_dt, np.asarray([[73151.25781719, -73201.64714963, -153600.],
                                                      [62111.25781719, -62150.84916874, 0.],
                                                      [-120741.66319236, 120710.84916874, 0.]]), 1e-1)

    def test_calculate_velocity(self):
        """
        Case in which kl and disp are identical for each component and equal to 1.
        """
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.2, 0.8]), n_points=4, p_in=5.0, temp=313,
                           c_len=1, u_in=1, void_frac=0.995, disp=[0, 0], kl=[5, 5], rho_p=2, append_helium=True,
                           dimensionless=False, p_out=5.0)
        solver = Solver(params)

        velocities = solver.calculate_velocities(np.asarray([[0.35, 0.65, 1], [0.1, 0.2, 1], [0.1, 0.3, 1]]),
                                                 np.asarray([[0.02, 0.05, 0], [0.2, 0.2, 0], [0.3, 0.3, 0]]),
                                                 np.asarray([[0.01, 0.03, 0], [0.05, 0.04, 0], [0.25, 0.25, 0]]))

        # The summed up term can be calculate using Wolfram Alpha with input:
        # Divide[\(40)1-0.6\(41),0.6]*2* ({{1,2},{3,4},{2,2}}-{{0.5,1.5},{2,3},{1,1}})-Divide[1,8.314*313]
        # {{-8,4,0},{4,-8,4},{0,8,-8}}{{0.3,0.7},{0.35,0.65},{0.4,0.6}}
        # Some parts need to be calculated with a separate matrix calculator
        np.testing.assert_allclose(velocities, np.asarray([-868.178, -1735.6]), rtol=1e-2)

    def test_solve_function(self):
        solver = self.initialize_solver()
        # ls = LinearizedSystem(solver, params)
        # ls.get_estimated_dt()
        p_partial_results = solver.solve()
        print(p_partial_results)

    def test_Linearized_Class(self):
        params = SysParams()
        params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.2, 0.8]), n_points=1000, p_in=5.0, temp=313,
                           c_len=1, u_in=1, void_frac=0.6, disp=[1, 1], kl=[1, 1], rho_p=500, p_out=5.0)
        solver = Solver(params)
        lin_sys = LinearizedSystem(solver, params)
        lin_sys.get_stiffness_estimate()
        lin_sys.get_estimated_dt()
