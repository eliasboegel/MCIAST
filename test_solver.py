import unittest

import numpy as np

from solver import *


class TestSolver(unittest.TestCase):

    def test_initialize_params_correct(self):
        params = SysParams(t_end=8, dt=0.001, y_in=np.asarray([0.2, 0.8]), n_points=11, p_in=5.0, p_out=5.0, temp=313,
                           c_len=4, u_in=2,
                           void_frac=0.6, disp=[16, 8], kl=[6, 8], p_he=12, rho_p=2)
        self.assertEqual(params.t_end, 4)
        self.assertEqual(params.dt, 0.0005)
        self.assertEqual(params.nt, 8000)
        self.assertEqual(params.p_in, 5)
        self.assertEqual(params.p_out, 5)
        self.assertEqual(params.n_points, 11)
        np.testing.assert_array_equal(params.y_in, np.asarray([0.2, 0.8]))
        self.assertEqual(params.temp, 313)
        self.assertEqual(params.void_frac, 0.6)
        self.assertEqual(params.rho_p, 2)
        self.assertEqual(params.p_he, 12)
        np.testing.assert_array_equal(params.p_partial_in, np.asarray([1, 4]))
        np.testing.assert_array_equal(params.kl, np.asarray([12, 16]))
        np.testing.assert_array_equal(params.disp, np.asarray([2, 1]))
        self.assertEqual(params.c_len, 1)
        self.assertEqual(params.dz, 0.1)
        self.assertEqual(params.dp_dz, 0)
        self.assertEqual(params.v_in, 1)
        self.assertEqual(params.n_components, 2)
        np.testing.assert_array_equal(params.p_total, np.asarray([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]))

    def test_initialize_solver(self):
        n_points = 11
        n_components = 2
        params = SysParams(t_end=8, dt=0.001, y_in=np.asarray([0.2, 0.8]), n_points=n_points, p_in=5.0, p_out=5.0, temp=313,
                           c_len=4, u_in=2,
                           void_frac=0.6, disp=16, kl=6, p_he=12, rho_p=2)
        solver = Solver(params)
        print("Matrix G: ", solver.g_matrix)
        print("Matrix L: ", solver.l_matrix)
        print("Matrix D: ", solver.d_matrix)
        print("Vector B: ", solver.b_vector)
        self.assertEqual(solver.g_matrix.shape, (n_points, n_points))
        self.assertEqual(solver.l_matrix.shape, (n_points, n_points))
        self.assertEqual(solver.d_matrix.shape, (n_points, n_components))
        self.assertEqual(solver.b_vector.shape, (n_points, ))

    def test_caclulate_dp_dt(self):
        params = SysParams(t_end=8, dt=0.001, y_in=np.asarray([0.2, 0.8]), n_points=3, p_in=5.0, p_out=5.0, temp=313,
                           c_len=4, u_in=2,
                           void_frac=0.6, disp=16, kl=6, p_he=12, rho_p=2)
        solver = Solver(params)

        dp_dt = solver.calculate_dp_dt([1, 2, 1], np.asarray([[0.3, 0.7], [0.35, 0.65]]), np.asarray([[1, 2], [3, 4], [2, 2]]),
                                       np.asarray([[0.5, 1.5], [2, 3], [1, 1]]))
        print("dp_dt: ", dp_dt)
