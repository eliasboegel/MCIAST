import scipy.sparse as sp
import scipy.optimize as opt
import iast
from mms import MMS
from plotter import *
from system_parameters import SysParams


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
        # Calculate the terms in the equation
        ldf = np.multiply(self.params.kl_matrix, q_eq - q_ads)
        lp = np.multiply(self.params.disp_matrix,
                         self.params.l_matrix.dot(p_partial))
        component_sums = np.sum(self.params.R * self.params.temp * self.params.void_frac_term * ldf - lp, axis=1)
        # Add up the terms in the equations for RHS
        if self.params.mms is True:
            rhs = -(component_sums + self.params.e_vector) / self.params.p_total - self.params.b_v_vector + \
                  self.MMS.S_nu
        else:
            rhs = -(component_sums + self.params.e_vector) / self.params.p_total - self.params.b_v_vector
        # Create LHS of the equation and solve
        lhs = self.params.g_matrix + self.params.f_matrix
        velocities = sp.linalg.lgmres(A=lhs, b=rhs, x0=np.ones(self.params.n_points - 1), atol=self.params.ls_error,
                                      maxiter=10 * (self.params.n_points - 1) ** 3)[0]
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
        # Calculate main terms
        advection_term = -self.params.g_matrix.dot(m_matrix)
        dispersion_term = np.multiply(self.params.disp_matrix, self.params.l_matrix.dot(p_partial))
        adsorption_term = -self.params.temp * self.params.R * self.params.void_frac_term * \
                          np.multiply(self.params.kl_matrix, q_eq - q_ads)
        # print("Advection term= :", advection_term)
        # print("Dispersion_term= ", dispersion_term)
        # print("Adsorpotion term= ", adsorption_term)
        # print("D term= ", self.params.d_matrix)
        # Add up main terms of the equation
        if self.params.mms is True:
            dp_dt = advection_term + dispersion_term + adsorption_term + self.params.d_matrix + self.MMS.S_pi

        else:
            dp_dt = advection_term + dispersion_term + adsorption_term + self.params.d_matrix

        return dp_dt

    def verify_pressures(self, p_partial):
        """
        Verify if all partial pressures sum up to total pressure.
        :param p_partial
        : Matrix containing partial pressures at each grid point
        :return: True if the pressures sum up to total pressure, false otherwise
        """
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
        # Compare the time gradient matrix to zero matrix
        return np.allclose(np.zeros(shape=(2 * self.params.n_points - 2, self.params.n_components)), du_dt,
                           atol=self.params.ls_error, rtol=0.0)

    def apply_iast(self, partial_pressures):
        """
        Applies the IAST method by calling the respective IAST solving functions.
        :param partial_pressures: Partial pressures of all the components, excluding the helium.
        :return: Equilibrium loadings for every component, excluding the helium (for which it is always 0).
        """
        equilibrium_loadings = np.zeros(partial_pressures.shape)
        zero = np.zeros(self.params.n_components - 1)
        # Apply IAST for each node
        for i in range(partial_pressures.shape[0]):
            # But only if partial pressures of components other than helium is not 0 (helium us last column)
            if np.allclose(partial_pressures[i, 0:-1], zero,
                           atol=self.params.ls_error) is False:
                # Pressure should be passed in bar to IAST
                equilibrium_loadings[i, 0:-1] = iast.solve(partial_pressures[i, 0:-1], self.params.isotherms)
        # print(f"Equilibrium loadings: {equilibrium_loadings}")
        return equilibrium_loadings

    def calculate_dudt(self, u, time):
        """
        Calculates the time derivative of the matrix u by calling multiple functions
        representing different steps of our algorithm.
        :param u: matrix u which is a concatenation of both the partial pressures and the adsorbed loadings.
        :param time: current point of time in the simulation, used for plotting.
        :return: Time derivative of the matrix u.
        """
        # print("u_old is:", u)
        # Disassemble solution matrix
        p_partial = u[:self.params.n_points - 1]
        q_ads = u[self.params.n_points - 1: 2 * self.params.n_points - 2]
        # Update source functions if MMS is used and get new loadings then
        if self.params.mms is True:
            self.MMS.update_source_functions(time)
            # print("S_pi matrix is:", self.MMS.S_pi)
            # print("S_nu matrix is:", self.MMS.S_nu)
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

    def solve(self, plot=True):
        """
        Performs the simulation.
        :param plot: True to turn on the plotting.
        :return: Matrix with final partial pressures and adsorbed loadings.
        """

        def crank_nicolson(u_new, u_old, time):
            """
                Represents Crank-Nicolson time-stepping method. Used in the non-linear
                solver to calculate the new u matrix.
            :param u_new: Approximation of the new matrix u.
            :param u_old: Old matrix u.
            :param time: Time in the simulation
            :return: Left-hand side of the Crank-Nicolson equation with all terms move to the left.
            """
            return u_old + 0.5 * self.params.dt * (self.calculate_dudt(u_new, time + self.params.dt) +
                                                   self.calculate_dudt(u_old, time)) - u_new

        def forward_euler(u_old, time):
            """
                Represents FE time-stepping method.
            :param u_old: Old matrix u.
            :param time: Time in the simulation.
            :return: New matrix u.
            """
            return u_old + self.params.dt * self.calculate_dudt(u_old, time)

        def backward_euler(u_new, u_old, time):
            """
                Represents Backward-Euler time-stepping method. Used in the non-linear
                solver to calculate the new u matrix.
            :param u_new: New matrix u.
            :param u_old: Old matrix u.
            :param time: Time in the simulation
            :return: Left-hand side of the Backward-Euler equation with all terms move to the left.
            """
            return u_old + self.params.dt * self.calculate_dudt(u_new, time + self.params.dt) - u_new

        # Create initial conditions, partial pressures for helium at the beginning are equal to total pressure
        q_ads_initial = np.zeros((self.params.n_points - 1, self.params.n_components))
        p_partial_initial = np.zeros((self.params.n_points - 1, self.params.n_components))
        if self.params.mms is False:
            p_partial_initial[:, -1] = self.params.p_total
        elif self.params.mms is True:
            self.MMS.update_source_functions(0)
            p_partial_initial = self.MMS.pi_matrix
        self.u_0 = np.concatenate((p_partial_initial, q_ads_initial), axis=0)
        # print("u_initial is ", self.u_0)

        t = 0
        du_dt = None
        self.u_1 = None

        if plot:
            plotter = Plotter(self)

        while (not self.check_steady_state(du_dt)) and t < self.params.t_end:
            # print(f"Another timestep...t={t}")
            # Get the solution
            if self.params.time_stepping == "BE":
                prediction = forward_euler(self.u_0, t)
                self.u_1 = opt.newton_krylov(lambda u: backward_euler(u, self.u_0, t), xin=prediction,
                                             f_tol=self.params.ls_error,
                                             maxiter=1000)
            elif self.params.time_stepping == "FE":
                self.u_1 = forward_euler(self.u_0, t)
            elif self.params.time_stepping == "CN":
                prediction = forward_euler(self.u_0, t)
                self.u_1 = opt.newton_krylov(lambda u: crank_nicolson(u, self.u_0, t), xin=prediction,
                                             f_tol=self.params.ls_error,
                                             maxiter=1000)
            # Check if solution makes sens
            # if not self.verify_pressures(self.u_1[0:self.params.n_points - 1]):
            #     print("The sum of partial pressures is not equal to 1!")

            if plot:
                plotter.plot(t)

                # plotter.plot_loadings(self.u_1[self.params.n_points - 1: 2 * self.params.n_points - 2])
            # Calculate derivative to check convergence
            du_dt = (self.u_1 - self.u_0) / self.params.dt

            # Update time
            t += self.params.dt

            # Initialize variables for the next time step
            self.u_0 = np.copy(self.u_1)

        # Return only partial pressures
        return self.u_1[0:self.params.n_points - 1]


def run_simulation():
    """
    Sets up and runs the simulation.
    """
    params = SysParams()
    params.init_params(t_end=30, dt=0.0001, y_in=np.asarray([0.36, 0.64]), n_points=10,
                       p_in=1e5, temp=313, c_len=1, u_in=1, void_frac=0.6, y_helium=0.0,
                       disp_helium=0.04, kl_helium=0, disp=[0.04, 0.04], kl=[5, 5],
                       rho_p=500, p_out=1e5, time_stepping="FE", dimensionless=True) #, mms=True,
                       # ms_pt_distribution="linear", mms_mode="transient", mms_convergence_factor=1 / 10)
    solver = Solver(params)
    p_partial_results = solver.solve()
    input()


if __name__ == "__main__":
    run_simulation()
