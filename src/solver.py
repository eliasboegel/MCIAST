import scipy.sparse as sp
import scipy.integrate as integrate
import iast
from plotter import *
from system_parameters import SysParams


class Solver:

    def __init__(self, sys_params):
        """
        Initalizes the solver with the initial conditions and parameters.
        :param sys_params: SysParams object containing all the system parameters.
        """
        self.params = sys_params

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
        if self.params.use_mms is True:
            rhs = -(component_sums + self.params.e_vector) / self.params.p_total - self.params.b_v_vector + \
                  self.params.MMS.S_nu
        else:
            rhs = -(component_sums + self.params.e_vector) / self.params.p_total - self.params.b_v_vector
        # Create LHS of the equation and solve
        lhs = self.params.g_matrix + self.params.f_matrix
        velocities = sp.linalg.lgmres(A=lhs, b=rhs, x0=np.ones(self.params.n_points - 1), atol=1e-16)[0]
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
        if self.params.use_mms is True:
            dp_dt = advection_term + dispersion_term + adsorption_term + self.params.d_matrix + self.params.MMS.S_pi

        else:
            dp_dt = advection_term + dispersion_term + adsorption_term + self.params.d_matrix

        return dp_dt

    def calculate_dq_ads_dt(self, q_eq, q_ads):
        """
        Calculates the time derivative of average component loadings in the adsorbent at each point.
        :param q_eq: Array containing equilibrium loadings of each component.
        :param q_ads: Array containing average component loadings in the adsorbent
        :return: Matrix containing time derivatives of average component loadings in the adsorbent at each point.
        """
        return np.multiply(self.params.kl_matrix, q_eq - q_ads)

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
                           atol=1e-16, rtol=0) is False:
                # Pressure should be passed in bar to IAST
                equilibrium_loadings[i, 0:-1] = iast.solve(partial_pressures[i, 0:-1], self.params.isotherms)
        # print(f"Equilibrium loadings: {equilibrium_loadings}")
        return equilibrium_loadings

    def calculate_dudt(self, t, u):
        """
        Calculates the time derivative of the matrix u by calling multiple functions
        representing different steps of our algorithm.
        :param u: matrix u which is a concatenation of both the partial pressures and the adsorbed loadings.
        :param t: current point of time in the simulation, used for plotting.
        :return: Time derivative of the matrix u.
        """
        # print("u_old is:", u)
        # Disassemble solution vector - first half of the vector is flattened partial pressures matrix, second is
        # flattened adsorbed loadings matrix
        p_partial = np.reshape(u[:self.params.n_components*(self.params.n_points - 1)],
                               (self.params.n_points - 1, self.params.n_components), "F")
        q_ads = np.reshape(u[self.params.n_components*(self.params.n_points - 1):],
                           (self.params.n_points - 1, self.params.n_components), "F")
        # Update source functions if MMS is used and get new loadings then
        if self.params.use_mms is True:
            self.params.MMS.update_source_functions(t)
            # print("S_pi matrix is:", self.MMS.S_pi)
            # print("S_nu matrix is:", self.MMS.S_nu)
            q_eq = self.params.MMS.q_eq_matrix
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
        # Assemble and return solution gradient 1D vector
        du_dt = np.concatenate((dp_dt.flatten("F"), dq_ads_dt.flatten("F")), axis=0)
        # print("du_dt matrix is:", du_dt)
        # print("Another internal iteration...")
        return du_dt

    def solve(self):
        """
        Performs the simulation.
        :return: A time vector of t_samples, a 2D array of p_i_evolution that stores the values of outlet pressures
        divided by inlet pressures over time, a 3D array of q_ads_evolution that stores q_ads matrices over time.
        The results do not contain the fill gas.
        """

        some_data = pd.read_csv("test_data/n2.csv", skiprows=1)
        co2_isotherm = pyiast.ModelIsotherm(some_data,loading_key="Loading(mmol/g)",pressure_key="P(bar)",model="Langmuir")
        n2_isotherm = pyiast.ModelIsotherm(some_data,loading_key="Loading(mmol/g)",pressure_key="P(bar)",model="Langmuir")
        co2_isotherm.params["M"] = 1
        co2_isotherm.params["K"] = 3.317e-4
        n2_isotherm.params["M"] = 0.3
        n2_isotherm.params["K"] = 1e-5
        self.params.isotherms = [co2_isotherm, n2_isotherm]
        
        # Run the scipy integrator
        sol = integrate.solve_ivp(fun=self.calculate_dudt, y0=self.params.u_0, t_span=(0.0, self.params.t_end),
                                  method=self.params.time_stepping_scheme, t_eval=self.params.t_samples,
                                  atol=self.params.atol, rtol=1e-13, vectorized=False)
        # Slice and process the results
        t_samples = sol.t
        # Create arrays to store them
        p_i_evolution = np.zeros((sol.t.shape[0], self.params.n_components-1))
        q_ads_evolution = np.zeros((sol.t.shape[0], self.params.n_points-1, self.params.n_components-1))
        # Get outlet pressures only. A row of p_i_evolution[t] are outlet pressures at time t.
        for i in range(0, self.params.n_components-1):
            p_i_evolution[:, i] = sol.y[(i + 1) * (self.params.n_points - 1) - 1]
        p_i_evolution /= self.params.p_partial_in[:-1]
        # Get q_ads matrices. q_ads_evolution[t] is q_ads matrix at time t.
        for i in range(0, t_samples.shape[0]):
            q_ads_evolution[i] = np.reshape(sol.y[self.params.n_components * (self.params.n_points - 1):
                                                  -(self.params.n_points - 1), i], (self.params.n_points - 1,
                                                                                    self.params.n_components-1), "F")
        return t_samples, p_i_evolution, q_ads_evolution


def run_simulation():
    """
    Sets up and runs the simulation and plots the results.
    """
    params = SysParams()
    params.init_params(t_end=2, atol=1e-6, dt=0.1, y_in=np.asarray([0.36, 0.64]), n_points=5,
                       p_in=1e5, temp=313, c_len=1, u_in=1, void_frac=0.6, y_fill_gas=0.0,
                       disp_fill_gas=0.04, kl_fill_gas=0, disp=[0.04, 0.04], kl=[5, 5],
                       rho_p=500, p_out=1e5, time_stepping_method="RK45", dimensionless=True)
    solver = Solver(params)
    t, p_i_evolution, q_ads_evolution = solver.solve()
    # print("time vector is:", t)
    # print("partial pressure evolution is:", p_i_evolution)
    # print("q_ads evolution is:", q_ads_evolution)
    plotter = Plotter(params)
    plotter.plot(t, p_i_evolution, q_ads_evolution)
    input()


if __name__ == "__main__":
    run_simulation()
