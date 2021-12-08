import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt
import pyiast
import pandas as pd
import os


class SysParams:
    def __init__(self):

        self.t_end = 0
        self.dt = 0
        self.nt = 0
        self.p_in = 0
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

    def init_params(self, y_in, n_points, p_in, temp, c_len, u_in, void_frac, disp, kl, rho_p, append_helium = True, t_end=40, dt=0.001, ):
        """
        Initializes the solver with the parameters that remain constant throughout the calculations
        and the initial conditions. The variables are turned into the dimensionless equivalents. The presence of helium
        is implicit. It means that it is always present no matter what parameters are passed. Its pressure is equal
        to the pressure of all components at the inlet. Therefore, the number of components is always len(y_in)+1.

        :param t_end: Final time point.
        :param dt: Length of one time step.
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
        """
        # Dimensionless points in time
        self.t_end = t_end * u_in / c_len
        self.dt = dt * u_in / c_len
        self.nt = self.t_end / self.dt
        # Pressure at the inlet is also equal to the total pressure at each point of the grid
        self.p_in = p_in
        self.p_total = p_in
        self.n_points = n_points
        if np.sum(y_in) != 1:
            raise Exception("Sum of mole fractions is not equal to 1")
        if temp < 0:
            raise Exception("Temperature cannot be below 0")
        self.temp = temp
        if not 0 <= void_frac <= 1:
            raise Exception("Void fraction is incorrect")
        self.void_frac = void_frac
        self.rho_p = rho_p

        # Dimensionless length of the column (always 1)
        self.c_len = 1
        self.dz = self.c_len / (n_points - 1)
        # Dimensionless gradient of the total pressure
        # This can be modified if we assume the gradient is not constant
        self.dp_dz = 0
        # Dimensionless inlet velocity (always 1)
        self.v_in = 1

        if append_helium:
            # 0 appended at the end for helium.
            self.y_in = np.append(np.asarray(y_in), 0)
            # Dimensionless mass transfer coefficients, with the coefficient of helium appended
            self.kl = np.append(np.asarray(kl) * c_len / u_in, 0)
            # Dimensionless dispersion coefficients, with the coefficient of helium appended
            self.disp = np.append(np.asarray(disp) / (c_len * u_in), 1)
        else:
            self.y_in = np.asarray(y_in)
            self.kl = np.asarray(kl) * c_len / u_in
            self.disp = np.asarray(disp) / (c_len * u_in)
        self.p_partial_in = self.y_in * p_in

        # The number of components assessed based on the length of y_in array (plus helium)
        self.n_components = len(self.y_in)

    def init_params_dimensionless(self, y_in, n_points, p_in, temp, void_frac, disp, kl, rho_p, append_helium=True, t_end=40, dt=0.001):
        """
        Initializes the solver with the parameters that remain constant throughout the calculations
        and the initial conditions. The variables are assumed to be dimensionless. The presence of helium
        is implicit. It means that it is always present no matter what parameters are passed. Its pressure is equal
        to the pressure of all components at the inlet. Therefore, the number of components is always len(y_in)+1.

                :param t_end: Dimensionless final time point.
                :param dt: Dimensionless length of one time step.
                :param y_in: Array containing mole fractions at the start.
                :param n_points: Number of grid points.
                :param p_in: Total pressure at the inlet.
                :param temp: Temperature of the system in Kelvins.
                :param void_frac: Void fraction (epsilon).
                :param disp: Array containing dimensionless dispersion coefficient for every component.
                :param kl: Array containing dimensionless effective mass transport coefficient of every component.
                :param rho_p: Density of the adsorbent.
                """
        self.t_end = t_end
        self.dt = dt
        self.nt = self.t_end / self.dt
        self.n_points = n_points
        self.rho_p = rho_p
        self.c_len = 1
        self.dz = self.c_len / (n_points - 1)
        self.dp_dz = 0
        self.v_in = 1
        self.n_components = 0

        if append_helium:
            self.y_in = np.append(np.asarray(y_in), 0)
            self.kl = np.append(np.asarray(kl), 0)
            self.disp = np.append(np.asarray(disp), 1)
        else:
            self.y_in = np.asarray(y_in)
            self.kl = np.asarray(kl)
            self.disp = np.asarray(disp)
        self.p_partial_in = self.y_in * p_in

        if temp < 0:
            raise Exception("Temperature cannot be below 0")
        self.temp = temp
        if not 0 <= void_frac <= 1:
            raise Exception("Void fraction is incorrect")
        self.void_frac = void_frac


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
        # Check if those sizes are correct (consult Jan)
        self.g_matrix = np.diag(np.full(self.params.n_points - 1, -1), -1) + np.diag(
            np.full(self.params.n_points - 1, 1), 1)
        self.g_matrix[self.g_matrix.shape[0] - 1][self.g_matrix.shape[1] - 3] = 1
        self.g_matrix[self.g_matrix.shape[0] - 1][self.g_matrix.shape[1] - 2] = -4
        self.g_matrix[self.g_matrix.shape[0] - 1][self.g_matrix.shape[1] - 1] = 3
        self.g_matrix = (1 / self.params.dz) ** 2 * self.g_matrix

        self.l_matrix = np.diag(np.full(self.params.n_points - 1, 1), -1) + np.diag(
            np.full(self.params.n_points - 1, 1), 1) + np.diag(np.full(self.params.n_points, -2), 0)
        self.l_matrix[self.l_matrix.shape[0] - 1][self.l_matrix.shape[1] - 2] = 2
        self.l_matrix[self.l_matrix.shape[0] - 1][self.l_matrix.shape[1] - 1] = -2
        self.l_matrix = (1 / self.params.dz) ** 2 * self.l_matrix

        self.d_matrix = np.zeros((self.params.n_points, self.params.n_components))
        first_row = (self.params.p_partial_in / (self.R * self.params.temp)) * (
                (self.params.v_in / (2 * self.params.dz)) + (self.params.disp / (self.params.dz ** 2)))
        self.d_matrix[0] = first_row  # idk if that works

        self.b_vector = np.zeros(self.params.n_points)
        self.b_vector[0] = - self.params.v_in / (2 * self.params.dz)

        # Dimensionless mass transfer coefficients matrix
        self.kl_matrix = np.broadcast_to(self.params.kl, (self.params.n_points, self.params.n_components))

        # Dimensionless dispersion coefficients matrix
        self.disp_matrix = np.broadcast_to(self.params.disp, (self.params.n_points, self.params.n_components))

    def calculate_velocities(self, p_partial, q_eq, q_ads):
        """
        Calculates velocities at all grid points. It is assumed that dpt/dxi = 0.
        :param p_partial: Matrix containing partial pressures of all the components at every grid point.
        Each row represents different grid point.
        :param p_total: Vector containing total pressures at each grid point.
        :param q_eq: Matrix containing equilibrium loadings of each component at each grid point.
        :param q_ads: Matrix containing average loadings in the

        :return: Array containing velocities at each grid point.
        """

        ldf = self.params.kl.T * (q_eq - q_ads)
        lp = (self.disp_matrix / (self.R * self.params.temp)) * self.l_matrix.dot(p_partial)
        void_frac_term = ((1 - self.params.void_frac) / self.params.void_frac) * self.params.rho_p
        component_sums = np.sum(void_frac_term * ldf - lp, axis=1)
        # Can we assume that to total pressure is equal to the sum of partial pressures in the beginning?
        # What about helium?
        # p_t = np.sum(p_partial, axis=1)
        rhs = -(1 / self.params.p_total) * self.R * self.params.temp * component_sums - self.b_vector
        lhs = self.g_matrix
        velocities = np.linalg.solve(lhs, rhs)
        return velocities

    def calculate_dp_dt(self, velocities, p_partial, q_eq, q_ads):
        """
        Calculates the time derivatives of partial pressures of each component for each grid point.
        Each row represents one grid point. Each column represents one component.
        :param velocities: Array containing velocities at each grid point.
        :param p_partial: Matrix containing partial pressures of every component at each grid point.
        :param q_eq: Matrix containing equilibrium loadings of every component at each grid point.
        :param q_ads: Matrix containing average loadings in the adsorbent of every component at each grid point.
        :param ro_p: Density of the adsorbent (?)
        :return: Matrix containing the time derivatives of partial pressures of each component at each grid point.
        """
        # This can be removed if we assume that a matrix in a correct form is passed
        velocities = velocities.reshape((-1, 1))

        m_matrix = np.multiply(velocities, p_partial)
        void_frac_term = ((1 - self.params.void_frac) / self.params.void_frac) * self.params.rho_p

        advection_term = -np.dot(self.g_matrix, m_matrix)
        dispersion_term = self.disp_matrix * np.dot(self.l_matrix, p_partial)
        adsorption_term = -self.params.temp * self.R * void_frac_term * self.kl_matrix * (
                q_eq - q_ads) + self.d_matrix

        dp_dt = advection_term + dispersion_term + adsorption_term
        return dp_dt

    def verify_pressures(self, p_partial):
        """
        Verify if all partial pressures sum up to 1.
        :param p_partial
        : Matrix containing partial pressures at each grid point
        :return: True if the pressures sum up to 1, false otherwise
        """
        p_summed = np.sum(p_partial, axis=1)
        return np.allclose(p_summed, self.params.p_total, 1.e-3, 0)

    def calculate_next_pressure(self, p_partial_old, dp_dt):
        """
        Steps in time to calculate the partial pressures of each component in the next point of time.
        :param p_partial_old: Matrix containing old partial pressures.
        :param dp_dt: Matrix containing time derivatives of partial pressures of each component at each grid point.
        :return: Matrix containing new partial pressures.
        """
        return p_partial_old + self.params.dt * dp_dt

    def calculate_dq_ads_dt(self, q_eq, q_ads):
        """
        Calculates the time derivative of average component loadings in the adsorbent at each point.
        :param q_eq: Array containing equilibrium loadings of each component.
        :param q_ads: Array containing average component loadings in the adsorbent
        :return: Matrix containing time derivatives of average component loadings in the adsorbent at each point.
        """
        dq_ads_dt = np.transpose(self.params.kl) * (q_eq - q_ads)
        return dq_ads_dt

    def calculate_next_q_ads(self, q_ads_old, dq_ads_dt):
        """
        Steps in time to calculate the average loadings in the adsorbent of each component in the next point of time.
        :param q_ads_old: Matrix containing old average loadings.
        :param dq_ads_dt: Matrix containing time derivatives of average loadings of each component at each grid point.
        :return: Matrix containing new average loadings.
        """
        return q_ads_old + self.params.dt * dq_ads_dt

    def check_equilibrium(self, p_partial_new):
        """
        Checks if the components have reached equilibrium and the adsorption process is finished by comparing
        partial pressures at the inlet and the outlet.
        :param p_partial_old: Matrix containing old partial pressures of all components at every point.
        :param p_partial_new: Matrix containing new partial pressures of all components at every point.
        :return: True if the equilibrium is reached, false otherwise.
        """
        if p_partial_new is None:
            return False
        return np.allclose(p_partial_new[0], p_partial_new[p_partial_new.shape[0] - 1], 1.e-5)

    def load_pyiast(self, partial_pressures):

        dirpath = os.path.abspath(os.path.dirname(__file__))

        df_N2 = pd.read_csv(dirpath + "/n2.csv", skiprows=1)
        N2_isotherm = pyiast.ModelIsotherm(df_N2, loading_key="Loading(mmol/g)", pressure_key="P(bar)",
                                           model="Langmuir")

        df_CO2 = pd.read_csv(dirpath + "/co2.csv", skiprows=1)
        CO2_isotherm = pyiast.ModelIsotherm(df_CO2, loading_key="Loading(mmol/g)", pressure_key="P(bar)",
                                            model="Langmuir")

        isotherms = np.array([
            [N2_isotherm.params['M'] * N2_isotherm.params['K'], N2_isotherm.params['K']],
            [CO2_isotherm.params['M'] * CO2_isotherm.params['K'], CO2_isotherm.params['K']]
        ])

        equilibrium_loadings = np.zeros(partial_pressures.shape)
        for i in range(1):
            equilibrium_loadings[i] = pyiast.iast(partial_pressures[i], [N2_isotherm, CO2_isotherm], warningoff=True)

        print(f"Equilibrium loadings: {equilibrium_loadings}")
        return equilibrium_loadings

    def solve(self):

        q_eq = np.zeros((self.params.n_points, self.params.n_components))
        q_ads = np.zeros((self.params.n_points, self.params.n_components))

        remaining_pressure = np.zeros((self.params.n_points - 1, self.params.n_components))
        remaining_pressure[:, -1] = self.params.p_total
        p_partial = np.vstack((self.params.p_partial_in, remaining_pressure))

        print(p_partial)
        t = 0

        while (not self.check_equilibrium(p_partial)) or t < self.params.t_end:

            # Calculate new loadings
            q_eq = self.load_pyiast(p_partial)  # Call the IAST (pyiast for now)
            dq_ads_dt = self.calculate_dq_ads_dt(q_eq, q_ads)
            q_ads = self.calculate_next_q_ads(q_ads, dq_ads_dt)
            # Calculate new velocity
            v = self.calculate_velocities(p_partial, q_eq, q_ads)

            # Calculate new partial pressures
            dp_dt = self.calculate_dp_dt(v, p_partial, q_eq, q_ads)
            p_partial = self.calculate_next_pressure(p_partial, dp_dt)
            if not self.verify_pressures(p_partial):
                print("The sum of partial pressures is not equal to 1!")

            # Add something for plotting here
            print(p_partial)

            t += self.params.dt

        return p_partial


class LinearizedSystem:
    def __init__(self, solver, sys_params):
        self.__solver = solver
        self.__params = sys_params

    def get_lin_sys_matrix(self, peclet_magnitude):
        # Calculate LHS matrix
        lhs = self.__solver.g_matrix + sp.diags(diagonals=self.__params.dp_dz/self.__params.p_total)
        # Calculate RHS matrix
        rhs = self.l_matrix.dot(self.__params.p_total / peclet_magnitude) / self.__params.p_total
        # Solve for nu approximation
        nu = sp.linalg.spsolve(lhs, rhs)
        # Create linearized system matrix
        a_matrix = -self.__solver.g_matrix * nu + self.l_matrix / peclet_magnitude
        return a_matrix

    def get_stiffness_estimate(self):
        for (peclet_number, peclet_magnitude) in (("largest Peclet number", 1 / np.min(self.params.disp)),
                                                  ("smallest Peclet number", 1 / np.max(self.params.disp)),
                                                  ("average Peclet number", 1 / np.mean(self.params.disp))):
            a_matrix = self.get_lin_sys_matrix(peclet_magnitude)
            lambda_max = sp.linalg.eigs(a_matrix, k=1, which="LM")
            lambda_min = sp.linalg.eigs(a_matrix, k=1, which="SM")
            stiffness = np.absolute(lambda_max/lambda_min)
            print(f"Stiffness of linearized system matrix for {peclet_number} is "
                  f"{self.linearized_system(peclet_magnitude)[1]}")

    def stability_analysis(self):
        a_matrix = get_lin_sys_matrix(self, 1 / np.max(self.params.disp))
        lambda_max = sp.linalg.eigs(a_matrix, k=1, which="LM")

        def rk4_stability_equation(u):
            return 1 + u + u ** 2 / 2 + u ** 3 / 6 + u ** 4 / 24

        def fe_stability_equation(u):
            return 1 / (1 - u)

        def stability_condition(f, dt):
            u = lambda_max * dt
            return np.absolute(f(u)) - 1

        for (f_name, f) in (("RK4", rk4_stability_equation), ("FE", fe_stability_equation)):
            dt = opt.fsolve(func=stability_condition, x0=np.array(0.0), maxfev=1000)
            print(f"Estimated timestep for stability for {f_name} is {dt} seconds")
