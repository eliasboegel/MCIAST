import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt
from src import iast
from src.mms import MMS


class Solver:

    def __init__(self, sys_params):
        """
        Initalizes the solver with the initial conditions and parameters.
        :param sys_params: SysParams object containing all the system parameters.
        """
        self.params = sys_params

        if self.params.mms is True:
            self.MMS = MMS(self.params)

    def calculate_velocities(self, p_partial, q_eq, q_ads):
        """
        Calculates velocities at all grid points. It is assumed that dpt/dxi = 0.
        :param p_partial: Matrix containing partial pressures of all the components at every grid point.
        Each row represents different grid point.
        :param q_eq: Matrix containing equilibrium loadings of each component at each grid point.
        :param q_ads: Matrix containing average loadings in the

        :return: Array containing velocities at each grid point.
        """
        # print(f"b_Vector: {self.b_vector}")
        # print(f"g_matrix {self.g_matrix}")
        # print(f"f_matrix: {self.f_matrix}")
        p_total = np.sum(p_partial, axis=1)
        ldf = np.multiply(self.params.kl_matrix, q_eq - q_ads)
        lp = np.multiply(self.params.disp_matrix / (self.params.R * self.params.temp),
                         self.params.l_matrix.dot(p_partial))
        component_sums = np.sum(self.params.void_frac_term * ldf - lp, axis=1)
        # Can we assume that to total pressure is equal to the sum of partial pressures in the beginning?
        # What about helium?
        # p_t = np.sum(p_partial, axis=1)
        if self.params.mms is True:
            rhs = -self.params.R * self.params.temp * component_sums / self.params.p_total - self.params.b_v_vector + \
                  self.MMS.S_nu

        else:
            rhs = -self.params.R * self.params.temp * component_sums / self.params.p_total - self.params.b_v_vector
        lhs = self.params.g_matrix + self.params.f_matrix
        velocities = sp.linalg.spsolve(lhs, rhs)
        return velocities

    def calculate_dp_dt(self, velocities, p_partial, q_eq, q_ads):
        """
        Calculates the time derivatives of partial pressures of each component for each grid point.
        Each row represents one grid point. Each column represents one component.
        :param velocities: Array containing velocities at each grid point.
        :param p_partial: Matrix containing partial pressures of every component at each grid point.
        :param q_eq: Matrix containing equilibrium loadings of every component at each grid point.
        :param q_ads: Matrix containing average loadings in the adsorbent of every component at each grid point.
        :return: Matrix containing the time derivatives of partial pressures of each component at each grid point.
        """
        # This can be removed if we assume that a matrix in a correct form is passed
        velocities = velocities.reshape((-1, 1))

        m_matrix = np.multiply(velocities, p_partial)

        advection_term = -self.params.g_matrix.dot(m_matrix)
        dispersion_term = np.multiply(self.params.disp_matrix, self.params.l_matrix.dot(p_partial))
        adsorption_term = -self.params.temp * self.params.R * self.params.void_frac_term * \
                          np.multiply(self.params.kl_matrix, q_eq - q_ads) + self.params.d_matrix

        if self.params.mms is True:
            dp_dt = advection_term + dispersion_term + adsorption_term + self.MMS.S_pi

        else:
            dp_dt = advection_term + dispersion_term + adsorption_term

        return dp_dt

    def verify_pressures(self, p_partial):
        """
        Verify if all partial pressures sum up to 1.
        :param p_partial
        : Matrix containing partial pressures at each grid point
        :return: True if the pressures sum up to 1, false otherwise
        """
        if self.params.mms is True:
            return np.allclose(np.sum(p_partial, axis=1), self.MMS.pt, atol=self.params.ls_error)
        else:
            return np.allclose(np.sum(p_partial, axis=1), self.params.p_total, atol=self.params.ls_error)

    def calculate_dq_ads_dt(self, q_eq, q_ads):
        """
        Calculates the time derivative of average component loadings in the adsorbent at each point.
        :param q_eq: Array containing equilibrium loadings of each component.
        :param q_ads: Array containing average component loadings in the adsorbent
        :return: Matrix containing time derivatives of average component loadings in the adsorbent at each point.
        """
        return np.multiply(self.params.kl_matrix, q_eq - q_ads)

    def check_steady_state(self, du_dt):
        """
        Checks if the components have reached equilibrium and the adsorption process is finished by comparing
        partial pressures at the inlet and the outlet.
        :param p_partial_old: Matrix containing old partial pressures of all components at every point.
        :param p_partial_new: Matrix containing new partial pressures of all components at every point.
        :return: True if the equilibrium is reached, false otherwise.
        """
        if du_dt is None:
            return False
        return np.allclose(np.zeros(shape=(2 * self.params.n_points - 2, self.params.n_components)), du_dt,
                           self.params.ls_error)

    def apply_iast(self, partial_pressures):
        equilibrium_loadings = np.zeros(partial_pressures.shape)
        zero = np.zeros(self.params.n_components - 1)
        for i in range(partial_pressures.shape[0]):
            if np.allclose(partial_pressures[i, 0:-1], zero,
                           atol=self.params.ls_error) is False:
                equilibrium_loadings[i, 0:-1] = iast.solve(partial_pressures[i, 0:-1] / 1e5, self.params.isotherms)
        # print(f"Equilibrium loadings: {equilibrium_loadings}")
        return equilibrium_loadings

    def calculate_dudt(self, u, time):
        # print("u_old is:", u)
        # Disassemble solution matrix
        p_partial = u[:self.params.n_points - 1]
        q_ads = u[self.params.n_points - 1: 2 * self.params.n_points - 2]
        # Update source functions if MMS is used and get new loadings then
        if self.params.mms is True:
            self.MMS.update_source_functions(time)
            q_eq = self.MMS.q_eq_matrix
        # Calculate new loadings
        else:
            q_eq = self.apply_iast(p_partial)  # Call the IAST
        # Calculate loading derivative
        # print("q_eq matrix is:", q_eq)
        dq_ads_dt = self.calculate_dq_ads_dt(q_eq, q_ads)
        # print("dq_ads_dt matrix is:", dq_ads_dt)
        # Calculate new velocity
        v = self.calculate_velocities(p_partial, q_eq, q_ads)
        # print("v vector is:", v)
        # Calculate new partial pressures derivative
        dp_dt = self.calculate_dp_dt(v, p_partial, q_eq, q_ads)
        # print("dp_dt matrix is:", dp_dt)
        # Assemble and return solution gradient matrix
        du_dt = np.concatenate((dp_dt, dq_ads_dt), axis=0)
        # print("du_dt matrix is:", du_dt)
        # print("Another internal iteration...")
        return du_dt

    def solve(self):
        def crank_nicolson(u_new, u_old):
            return u_old + 0.5 * self.params.dt * (self.calculate_dudt(u_new, t + self.params.dt) +
                                                   self.calculate_dudt(u_old, t)) - u_new

        def forward_euler(u_old):
            return u_old + self.params.dt * self.calculate_dudt(u_old, t)

        def backward_euler(u_new, u_old):
            return u_old + self.params.dt * self.calculate_dudt(u_new, t + self.params.dt) - u_new

        # Create initial conditions
        q_ads_initial = np.zeros((self.params.n_points - 1, self.params.n_components))
        p_partial_initial = np.zeros((self.params.n_points - 1, self.params.n_components))
        p_partial_initial[:, -1] = self.params.p_total
        u_0 = np.concatenate((p_partial_initial, q_ads_initial), axis=0)
        print("u_initial is ", u_0)

        t = 0
        du_dt = None
        u_1 = None

        while (not self.check_steady_state(du_dt)) and t < self.params.t_end:
            print("Another timestep...")
            if self.params.time_stepping == "BE":
                u_1 = opt.newton_krylov(lambda u: backward_euler(u, u_0), xin=u_0, f_tol=self.params.ls_error)
            elif self.params.time_stepping == "FE":
                u_1 = forward_euler(u_0)
            elif self.params.time_stepping == "CN":
                u_1 = opt.newton_krylov(lambda u: crank_nicolson(u, u_0), xin=u_0, f_tol=self.params.ls_error)

            if not self.verify_pressures(u_1[0:self.params.n_points - 1]):
                print("The sum of partial pressures is not equal to 1!")

            # Calculate derivative to check convergance
            du_dt = (u_1 - u_0) / self.params.dt

            # Update time
            t += self.params.dt

            # Initialize variables for the next time step
            u_0 = np.copy(u_1)

        return u_1[0:self.params.n_points - 1]
