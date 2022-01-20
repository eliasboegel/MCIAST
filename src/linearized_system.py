import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt
from src.system_parameters import SysParams


class LinearizedSystem:
    def __init__(self, sys_params):
        self.params = sys_params

    def get_lin_sys_matrix(self, disp_magnitude):
        a_matrix = -self.params.g_matrix + self.params.l_matrix * disp_magnitude
        return a_matrix

    def get_stiffness_estimate(self):
        disp_magnitude = np.mean(self.params.disp)
        a_matrix = self.get_lin_sys_matrix(disp_magnitude)
        lambda_max = sp.linalg.eigs(a_matrix, k=1, which="LM", return_eigenvectors=False, maxiter=100000, tol=1e-15)[0]
        lambda_min = sp.linalg.eigs(a_matrix, k=1, which="SM", return_eigenvectors=False, maxiter=100000, tol=1e-15)[0]
        stiffness = np.absolute(lambda_max / lambda_min)
        print(f"Stiffness of linearized system matrix for is {stiffness}")

    def get_estimated_dt(self):
        a_matrix = self.get_lin_sys_matrix(np.max(self.params.disp))
        lambda_max = (sp.linalg.eigs(a_matrix, k=1, which="LM", return_eigenvectors=False)[0])

        def rk4_stability_equation(u):
            return 1 + u + (u ** 2) / 2 + (u ** 3) / 6 + (u ** 4) / 24

        def be_stability_equation(u):
            return 1 / (1 - u)

        def fe_stability_equation(u):
            return 1 + u

        def stability_condition(dt, f):
            u = lambda_max * dt
            return np.absolute(f(u)) - 1

        for (f_name, f) in (("RK4", rk4_stability_equation), ("BE", be_stability_equation),
                            ("FE", fe_stability_equation)):
            dt = opt.fsolve(func=stability_condition, args=f, x0=np.array(2.0), maxfev=10000)[0]
            print(f"Estimated timestep for stability for {f_name} is {dt} seconds")


params = SysParams()
params.init_params(t_end=8, dt=0.001, y_in=np.asarray([0.25, 0.25, 0.25]), n_points=1000, p_in=2.00 * 1e5,
                   p_out=2.00 * 1e5, y_helium=0.25, disp_helium=0.004, kl_helium=1, temp=288, c_len=1, u_in=1,
                   void_frac=0.1, disp=[0.004, 0.004, 0.004], kl=[1, 1, 1], rho_p=1000)
linsys = LinearizedSystem(params)
linsys.get_stiffness_estimate()
linsys.get_estimated_dt()
