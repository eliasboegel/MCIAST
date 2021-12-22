import numpy as np
import scipy as sp
import scipy.optimize as opt

class LinearizedSystem:
    def __init__(self, solver, sys_params):
        self.solver = solver
        self.params = sys_params

    def get_lin_sys_matrix(self, peclet_magnitude):
        # Calculate LHS matrix
        # lhs = self.solver.g_matrix + self.solver.f_matrix
        # Calculate RHS matrix
        # rhs = self.solver.l_matrix.dot(self.params.p_total / peclet_magnitude) / self.params.p_total
        # Solve for nu approximation
        # nu = sp.linalg.spsolve(lhs, rhs)
        # Create linearized system matrix
        # a_matrix = -self.solver.g_matrix.multiply(nu.reshape((-1, 1))) + self.solver.l_matrix / peclet_magnitude
        a_matrix = -self.solver.g_matrix + self.solver.l_matrix / peclet_magnitude
        return a_matrix

    def get_stiffness_estimate(self):
        for (peclet_number, peclet_magnitude) in (("largest Peclet number", 1 / np.min(self.params.disp)),
                                                  ("smallest Peclet number", 1 / np.max(self.params.disp)),
                                                  ("average Peclet number", 1 / np.mean(self.params.disp))):
            a_matrix = self.get_lin_sys_matrix(peclet_magnitude)
            lambda_max = sp.linalg.eigs(a_matrix, k=1, which="LM", return_eigenvectors=False)[0]
            lambda_min = sp.linalg.eigs(a_matrix, k=1, which="SM", return_eigenvectors=False)[0]
            stiffness = np.absolute(lambda_max / lambda_min)
            print(f"Stiffness of linearized system matrix for {peclet_number} is {stiffness}")

    def get_estimated_dt(self):
        a_matrix = self.get_lin_sys_matrix(1 / np.max(self.params.disp))
        lambda_max = sp.linalg.eigs(a_matrix, k=1, which="LM", return_eigenvectors=False)[0]

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
            dt = opt.fsolve(func=stability_condition, args=f, x0=np.array(1.0), maxfev=10000)[0]
            print(f"Estimated timestep for stability for {f_name} is {dt} seconds")
