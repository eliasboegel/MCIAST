import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt
import iast
import os


class SysParams:
    def __init__(self):
        self.t_end = 0
        self.dt = 0
        self.nt = 0
        self.p_in = 0
        self.p_out = 0
        self.n_points = 0
        self.y_in = 0
        self.temp = 0
        self.void_frac = 0
        self.rho_p = 0
        self.kl = 0
        self.disp = 0
        self.c_len = 0
        self.dz = 0
        self.dp_dz = 0
        self.v_in = 0
        self.n_components = 0
        self.p_total = 0
        self.p_partial_in = 0
        self.mms = 0
        self.mms_mode = 0
        self.mms_conv_factor = 0
        self.ms_pt_distribution = 0
        self.outlet_boundary_type = 0
        self.void_frac_term = 0
        self.dis_error = 0
        self.ls_error = 0
        self.time_stepping = 0
        self.isotherms = 0

    def init_params(self, y_in, n_points, p_in, p_out, temp, c_len, u_in, void_frac, disp, kl, rho_p, dispersion_helium,
                    t_end=40, dt=0.001, time_stepping="BE", dimensionless=True, mms=False,
                    ms_pt_distribution="linear", mms_mode="transient", mms_convergence_factor=1000):

        """
        Initializes the solver with the parameters that remain constant throughout the calculations
        and the initial conditions. Depending on the dimensionless paramter, the variables might turned into the
        dimensionless equivalents. The presence of helium is implicit. It means that it is always present no matter
        what parameters are passed. Its pressure is equal to the pressure of all components at the inlet.
        Therefore, the number of components is always len(y_in)+1.

        :param p_out: Total pressure at the outlet.
        :param t_end: Final time point.
        :param dt: Length of one time step.
        :param dimensionless: Boolean that specifies whether dimensionless numbers are used.
        :param time_stepping: String that specifies the time of stepping methods.
        :param y_in: Array containing mole fractions at the start.
        :param n_points: Number of grid points.
        :param p_in: Total pressure at the inlet.
        :param temp: Temperature of the system in Kelvins.
        :param c_len: Column length.
        :param u_in: Speed at the inlet.
        :param void_frac: Void fraction (epsilon).
        :param disp: Array containing dispersion coefficient for every component.
        :param kl: Array containing effective mass transport coefficient of every component.
        :param rho_p: Density of the adsorbent.
        :param mms: Choose if dynamic code testing is switched on.
        :param ms_pt_distribution: Choose total pressure distribution for dynamic code testing.
        :param mms_mode: Choose if MMS is to be used to steady state or transient simulation.
        :param mms_convergence_factor: Choose how quickly MMS is supposed to reach steady state.
        """

        y_in = np.append(y_in, 0)
        kl = np.append(kl, 0)
        disp = np.append(disp, dispersion_helium)

        if dimensionless:
            # Dimensionless points in time
            self.t_end = t_end * u_in / c_len
            self.dt = dt * u_in / c_len
            self.nt = self.t_end / self.dt
            self.y_in = np.asarray(y_in)
            self.kl = np.asarray(kl) * c_len / u_in
            self.disp = np.asarray(disp) / (c_len * u_in)

        else:
            self.t_end = t_end
            self.dt = dt
            self.nt = self.t_end / self.dt
            self.y_in = np.asarray(y_in)
            self.kl = np.asarray(kl)
            self.disp = np.asarray(disp)

        # Pressure at the inlet is also equal to the total pressure at each point of the grid
        self.p_in = p_in
        self.p_out = p_out
        self.n_points = n_points
        self.p_total = np.linspace(p_in, p_out, n_points)[1:]

        if np.sum(y_in) != 1:
            raise Exception("Sum of mole fractions is not equal to 1")
        if temp < 0:
            raise Exception("Temperature cannot be below 0")
        self.temp = temp
        if not 0 <= void_frac <= 1:
            raise Exception("Void fraction is incorrect")
        self.void_frac = void_frac
        self.rho_p = rho_p
        # Used to calculate velocity
        self.void_frac_term = ((1 - self.void_frac) / self.void_frac) * self.rho_p

        # Dimensionless length of the column (always 1)
        self.c_len = 1
        self.dz = self.c_len / (n_points - 1)
        # Dimensionless gradient of the total pressure
        # This can be modified if we assume the gradient is not constant
        self.dp_dz = (p_out - p_in) / self.c_len
        # Dimensionless inlet velocity (always 1)
        self.v_in = 1

        self.p_partial_in = self.y_in * p_in

        # The number of components assessed based on the length of y_in array
        self.n_components = self.y_in.shape[0]

        # Parameters for outlet boundary condition
        if self.p_in == self.p_out:
            self.outlet_boundary_type = "Neumann"
        elif self.p_in != self.p_out:
            self.outlet_boundary_type = "Numerical"
        elif self.outlet_boundary_type != "Neumann" and self.outlet_boundary_type != "Numerical":
            raise Warning("Outlet boundary condition needs to be either Neumann or Numerical")

        # Determine the magnitude of errors
        self.time_stepping = time_stepping
        if self.time_stepping == "BE" or self.time_stepping == "FE":
            self.dis_error = max(self.dz**2, self.dt**2)
        elif self.time_stepping == "CN":
            self.dis_error = max(self.dz**2, self.dt**3)
        else:
            raise Warning("Only FE, BE, CN methods can be used!")
        self.ls_error = self.dis_error / 100

        # Parameters for running dynamic code verification using MMS
        self.mms = mms
        if self.mms is True:
            self.mms_mode = mms_mode
            self.mms_conv_factor = mms_convergence_factor
            self.ms_pt_distribution = ms_pt_distribution
            if dimensionless is False:
                raise Warning("mms must be initialized with dimensionless set to True")
        if type(self.mms) != bool:
            raise Warning("mms must be either True or False")

        dirpath = os.path.abspath(os.path.dirname(__file__))
        self.isotherms = iast.fit([dirpath + "/test_data/n2.csv", dirpath + "/test_data/co2.csv"])


class MMS:
    def __init__(self, sys_params, solver):
        # Initialize other classes as attributes
        self.__params = sys_params
        self.__solver = solver
        # Initialize 1D dummy arrays as attributes
        self.S_nu = np.zeros(self.__params.n_points - 1)
        self.pi = np.zeros(self.__params.n_points - 1)
        self.dpi_dz = np.zeros(self.__params.n_points - 1)
        self.d2pi_dz2 = np.zeros(self.__params.n_points - 1)
        self.dpi_dt = np.zeros(self.__params.n_points - 1)
        self.nu = np.zeros(self.__params.n_points - 1)
        self.dnu_dz = np.zeros(self.__params.n_points - 1)
        self.dnupi_dz = np.zeros(self.__params.n_points - 1)
        self.q_eq = np.zeros(self.__params.n_points - 1)
        self.q_ads = np.zeros(self.__params.n_points - 1)
        self.xi = np.linspace(0, self.__params.c_len, self.__params.n_points)[1:]
        # Initialize 2D arrays as attributes
        self.S_pi = np.zeros((self.__params.n_points - 1, self.__params.n_components))
        self.pi_matrix = np.zeros((self.__params.n_points - 1, self.__params.n_components))
        self.dpi_dz_matrix = np.zeros((self.__params.n_points - 1, self.__params.n_components))
        self.d2pi_dz2_matrix = np.zeros((self.__params.n_points - 1, self.__params.n_components))
        self.dpi_dt_matrix = np.zeros((self.__params.n_points - 1, self.__params.n_components))
        self.dnupi_dz_matrix = np.zeros((self.__params.n_points - 1, self.__params.n_components))
        self.q_eq_matrix = np.zeros((self.__params.n_points - 1, self.__params.n_components))
        self.q_ads_matrix = np.zeros((self.__params.n_points - 1, self.__params.n_components))
        self.delta_q_matrix = np.zeros((self.__params.n_points - 1, self.__params.n_components))
        self.pi_diffusion_matrix = np.zeros((self.__params.n_points - 1, self.__params.n_components))
        self.a = 0
        self.b = 0
        # Initialize constants as attributes
        if self.__params.mms_mode == "steady":
            self.b = 0
        elif self.__params.mms_mode == "transient":
            self.b = 1
        else:
            raise Warning("mms_mode needs to be either transient or steady!")
        if self.__params.ms_pt_distribution == "constant":
            self.a = 0
        elif self.__params.ms_pt_distribution == "linear":
            self.a = 1
        else:
            raise Warning("ms_pt_distribution needs to be either constant or linear!")
        self.pt = 1 - self.a * self.xi / 2
        self.pi_0 = self.pt[0] / self.__params.n_components
        self.nu_0 = 1
        self.dpt_dz = - self.a / 2
        self.c = self.__params.mms_conv_factor
        self.t_factor = 0
        self.RT = self.__solver.R * self.__params.temp
        self.void_term = (1 - self.__params.void_frac) / self.__params.void_frac * self.__params.rho_p * self.RT

    def calculate_pi_ms(self, i):
        self.pi = self.pi_0 + (-1) ** i * (np.sin(np.pi / 2 * self.xi) +
                                           self.b * self.t_factor * np.sin(np.pi * self.xi))

    def calculate_nu_ms(self):
        self.nu = 1 - 0.5 * np.sin(np.pi / 2 * self.xi) + self.b * self.t_factor * np.sin(np.pi * self.xi) ** 2

    def calculate_q_ads_ms(self, tau, i):
        self.q_ads = self.b * self.__params.kl[i] * self.xi * tau * self.t_factor

    def calculate_q_eq_ms(self, tau):
        self.q_eq = self.b * self.xi * (self.t_factor - tau * self.t_factor / self.c) + self.q_ads

    def calculate_dpi_dz_ms(self, i):
        self.dpi_dz = -self.a / (2 * self.__params.n_components) + \
                      (-1) ** i * np.pi * (0.5 * np.cos(np.pi / 2 * self.xi) +
                                           self.b * self.t_factor * np.cos(np.pi * self.xi))

    def calculate_d2pi_dz2_ms(self, i):
        self.d2pi_dz2 = -(-1) ** i * np.pi ** 2 * (0.25 * np.sin(np.pi / 2 * self.xi) +
                                                   self.b * self.t_factor * np.sin(np.pi * self.xi))

    def calculate_dpi_dt_ms(self, i):
        self.dpi_dt = self.b * (-(-1) ** i * self.t_factor * np.sin(np.pi * self.xi)) / self.c

    def calculate_dnu_dz_ms(self):
        self.dnu_dz = np.pi * (-0.25 * np.cos(np.pi / 2 * self.xi) +
                               self.b * 2 * self.t_factor * np.sin(np.pi * self.xi) * np.cos(np.pi * self.xi))

    def calculate_dnupi_dz(self):
        self.dnupi_dz = self.pi * self.dnu_dz + self.nu * self.dpi_dz

    def update_source_functions(self, tau):
        # Update time factor
        self.t_factor = np.e ** (-tau / self.c)
        # Update MS velocity for all components
        self.calculate_nu_ms()
        # Update 1st derivative of MS velocity for all components
        self.calculate_dnupi_dz()
        for i in range(0, self.__params.n_components):
            # Update MS pressure for component i
            self.calculate_pi_ms(i)
            self.pi_matrix[:, i] = np.copy(self.pi)
            # Update 1st derivative of MS pressure for component i
            self.calculate_dpi_dz_ms(i)
            self.dpi_dz_matrix[:, i] = np.copy(self.dpi_dz)
            # Update 2nd derivative of MS pressure for component i
            self.calculate_d2pi_dz2_ms(i)
            self.d2pi_dz2_matrix[:, i] = np.copy(self.d2pi_dz2)
            # Update the 1st derivative of nu*p_i for component i
            self.calculate_dnupi_dz()
            self.dnupi_dz_matrix[:, i] = np.copy(self.dnupi_dz)
            # Update MS time derivative of p_i
            self.calculate_dpi_dt_ms(i)
            self.dpi_dt_matrix[:, i] = np.copy(self.dpi_dt)
            # Update equivalent adsorbed loading
            self.calculate_q_ads_ms(tau, i)
            self.q_ads_matrix[:, i] = np.copy(self.q_ads)
            # Update equivalent loading
            self.calculate_q_eq_ms(tau)
            self.q_eq_matrix[:, i] = np.copy(self.q_eq)
        self.delta_q_matrix = self.q_eq_matrix - self.q_ads_matrix
        self.pi_diffusion_matrix = self.d2pi_dz2_matrix * self.__solver.disp_matrix
        self.S_pi = self.dpi_dt_matrix + self.dnupi_dz_matrix - self.pi_diffusion_matrix + \
                    self.void_term * self.__solver.kl_matrix * self.delta_q_matrix
        self.S_nu = self.dnu_dz + np.sum((self.void_term * self.__solver.kl_matrix *
                                          self.delta_q_matrix - self.pi_diffusion_matrix), axis=1) / self.pt


class OoA:
    def __init__(self, which, n, dt):
        self.n = n
        self.dt = dt
        self.type = which
        if self.type != "Space" and self.type != "Time":
            raise Warning("which of OoM must be either Space or Time")

    def analysis(self):
        discretization_list = [(self.dt, self.n), (self.dt / 2, 2 * self.n), (self.dt / 3, 3 * self.n)]
        error_list = []
        ss_params = SysParams()
        for (dt, nodes) in discretization_list:
            ss_params.init_params(t_end=10000, dt=dt, y_in=np.asarray([0.5, 0.5]), n_points=nodes, p_in=2e5,
                                  temp=298, c_len=1, u_in=1, void_frac=0.995, disp=[0.004, 0.004], kl=[4.35, 1.47],
                                  rho_p=1000, p_out=2e5, time_stepping="BE", dimensionless=True,
                                  dispersion_helium=0.004, mms=True, ms_pt_distribution="linear",
                                  mms_mode="steady", mms_convergence_factor=1000)
            ss_solver = Solver(ss_params)
            error_matrix = None
            if self.type == "Space":
                ss_solver.MMS.update_source_functions(0)
                dp_dt_manufactured = ss_solver.MMS.dpi_dt_matrix

                q_ads = ss_solver.MMS.q_ads_matrix
                p_partial = ss_solver.MMS.pi_matrix
                u = np.concatenate((p_partial, q_ads), axis=0)
                du_dt_calc = ss_solver.calculate_dudt(u, 0)
                dp_dt_calc = du_dt_calc[0:nodes - 1]

                error_matrix = dp_dt_calc - dp_dt_manufactured

            elif self.type == "Time":
                t_params = SysParams()
                t_params.init_params(t_end=10000, dt=dt, y_in=np.asarray([0.5, 0.5]), n_points=nodes, p_in=2e5,
                                     temp=298, c_len=1, u_in=1, void_frac=0.995, disp=[0.004, 0.004], kl=[4.35, 1.47],
                                     rho_p=1000, p_out=2e5, time_stepping="BE", dimensionless=True,
                                     dispersion_helium=0.004, mms=True, ms_pt_distribution="linear",
                                     mms_mode="transient", mms_convergence_factor=1000)
                t_solver = Solver(t_params)

                p_i_calc = t_solver.solve()

                ss_solver.MMS.update_source_functions(0)
                p_i_manufactured = ss_solver.MMS.pi_matrix

                error_matrix = p_i_calc - p_i_manufactured

            error_norm_i = np.sqrt(np.mean(error_matrix ** 2, axis=0))
            error_norm = np.sqrt(np.mean(error_norm_i ** 2))
            error_list.append(error_norm)

        OoA = np.log((error_list[2] - error_list[1]) / (error_list[1] - error_list[0])) / np.log(2)
        return OoA, discretization_list


class Solver:
    # Gas constant
    R = 8.314

    def __init__(self, sys_params):
        """
        Initalizes the solver with the initial conditions and parameters.
        :param sys_params: SysParams object containing all the system parameters.
        """
        self.params = sys_params

        # Those matrices will be used in the solver, their internal structure is fully explained in the report

        # Dimensionless mass transfer coefficients matrix
        self.kl_matrix = np.broadcast_to(self.params.kl, (self.params.n_points - 1, self.params.n_components))
        print("kl_matrix is", self.kl_matrix)

        # Dimensionless dispersion coefficients matrix
        self.disp_matrix = np.broadcast_to(self.params.disp, (self.params.n_points - 1, self.params.n_components))
        print("disp_matrix is", self.disp_matrix)

        # System matrices
        self.g_matrix = np.diag(np.full(self.params.n_points - 2, -1.0), -1) + np.diag(
            np.full(self.params.n_points - 2, 1.0), 1)
        self.g_matrix[-1, -3] = 1.0
        self.g_matrix[-1, -2] = -4.0
        self.g_matrix[-1, -1] = 3.0
        print("initial g_matrix is", self.g_matrix)
        self.g_matrix = self.g_matrix / (2.0 * self.params.dz)
        self.g_matrix = sp.csr_matrix(self.g_matrix)

        self.f_matrix = np.diag(self.params.dp_dz / self.params.p_total)
        # # print(f"f_matrix: {self.f_matrix}")
        self.f_matrix = sp.csr_matrix(self.f_matrix)
        # # print(f"f_matrix: {self.f_matrix.toarray()}")

        self.l_matrix = np.diag(np.full(self.params.n_points - 2, 1.0), -1) + np.diag(
            np.full(self.params.n_points - 2, 1.0), 1) + np.diag(np.full(self.params.n_points - 1, -2.0), 0)
        if self.params.outlet_boundary_type == "Neumann":
            self.l_matrix[-1, -2] = 2.0
        elif self.params.outlet_boundary_type == "Numerical":
            self.l_matrix[-1, -4] = -1.0
            self.l_matrix[-1, -3] = 4.0
            self.l_matrix[-1, -2] = -5.0
            self.l_matrix[-1, -1] = 2.0
        print("initial l_matrix is", self.l_matrix)
        self.l_matrix /= self.params.dz ** 2
        self.l_matrix = sp.csr_matrix(self.l_matrix)
        # print(f"l_matrix {self.l_matrix.toarray()}")

        if self.params.mms is True:
            self.MMS = MMS(self.params, self)
            p_partial_in = self.MMS.pi_0
            v_in = self.MMS.nu_0

        else:
            p_partial_in = self.params.p_partial_in
            v_in = self.params.v_in

        self.d_matrix = np.zeros((self.params.n_points - 1, self.params.n_components), dtype="float")
        first_row = p_partial_in * (
                (v_in / (2 * self.params.dz)) + (self.params.disp / (self.params.dz ** 2)))
        self.d_matrix[0] = first_row
        # self.d_matrix = sp.csr_matrix(self.d_matrix)

        self.b_v_vector = np.zeros(self.params.n_points - 1)
        # print(self.b_vector)
        self.b_v_vector[0] = - self.params.v_in / (2 * self.params.dz)

    def calculate_velocities(self, p_partial, q_eq, q_ads):
        """
        Calculates velocities at all grid points. It is assumed that dpt/dxi = 0.
        :param p_partial: Matrix containing partial pressures of all the components at every grid point.
        Each row represents different grid point.
        :param q_eq: Matrix containing equilibrium loadings of each component at each grid point.
        :param q_ads: Matrix containing average loadings in the

        :return: Array containing velocities at each grid point.
        """
        #print(f"b_Vector: {self.b_vector}")
        #print(f"g_matrix {self.g_matrix}")
        #print(f"f_matrix: {self.f_matrix}")
        p_total = np.sum(p_partial, axis=1)
        ldf = np.multiply(self.kl_matrix, q_eq - q_ads)
        lp = np.multiply(self.disp_matrix / (self.R * self.params.temp), self.l_matrix.dot(p_partial))
        component_sums = np.sum(self.params.void_frac_term * ldf - lp, axis=1)
        # Can we assume that to total pressure is equal to the sum of partial pressures in the beginning?
        # What about helium?
        # p_t = np.sum(p_partial, axis=1)
        if self.params.mms is True:
            rhs = -self.R * self.params.temp * component_sums / self.params.p_total - self.b_v_vector + \
                  self.MMS.S_nu

        else:
            rhs = -self.R * self.params.temp * component_sums / self.params.p_total - self.b_v_vector
        lhs = self.g_matrix + self.f_matrix
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

        advection_term = -self.g_matrix.dot(m_matrix)
        dispersion_term = np.multiply(self.disp_matrix, self.l_matrix.dot(p_partial))
        adsorption_term = -self.params.temp * self.R * self.params.void_frac_term * \
                          np.multiply(self.kl_matrix, q_eq - q_ads) + self.d_matrix

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
        return np.multiply(self.kl_matrix, q_eq - q_ads)

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
        zero = np.zeros(self.params.n_components-1)
        for i in range(partial_pressures.shape[0]):
            if np.allclose(partial_pressures[i, 0:-1], zero,
                           atol=self.params.ls_error) is False:
                equilibrium_loadings[i, 0:-1] = iast.solve(partial_pressures[i, 0:-1]/1e5, self.params.isotherms)
        #print(f"Equilibrium loadings: {equilibrium_loadings}")
        return equilibrium_loadings

    def calculate_dudt(self, u, time):
        #print("u_old is:", u)
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
        #print("q_eq matrix is:", q_eq)
        dq_ads_dt = self.calculate_dq_ads_dt(q_eq, q_ads)
        #print("dq_ads_dt matrix is:", dq_ads_dt)
        # Calculate new velocity
        v = self.calculate_velocities(p_partial, q_eq, q_ads)
        #print("v vector is:", v)
        # Calculate new partial pressures derivative
        dp_dt = self.calculate_dp_dt(v, p_partial, q_eq, q_ads)
        #print("dp_dt matrix is:", dp_dt)
        # Assemble and return solution gradient matrix
        du_dt = np.concatenate((dp_dt, dq_ads_dt), axis=0)
        #print("du_dt matrix is:", du_dt)
        #print("Another internal iteration...")
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

            if not self.verify_pressures(u_1[0:self.params.n_points-1]):
                print("The sum of partial pressures is not equal to 1!")

            # Calculate derivative to check convergance
            du_dt = (u_1 - u_0) / self.params.dt

            # Update time
            t += self.params.dt

            # Initialize variables for the next time step
            u_0 = np.copy(u_1)

        return u_1[0:self.params.n_points - 1]


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


ooa = OoA(which="Space", n=100, dt=0.001)
print(ooa.analysis()[0])
