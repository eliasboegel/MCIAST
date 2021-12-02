import numpy as np


class SysParams:
    def __init__(self, y_in, n_points, p_in, p_out, temp, c_len, u_in, void_frac, disp, kl, p_he, rho_p, t_end=40,
                 dt=0.001):
        """
        Initializes the solver with the parameters that remain constant throughout the calculations
        and the initial conditions. The variables are turned into the dimensionless equivalents.

        :param t_end: Final time point.
        :param dt: Length of one time step.
        :param y_in: Array containing mole fractions at the start.
        :param n_points: Number of grid points.
        :param p_in: Total pressure at the inlet.
        :param p_out: Total pressure at the outlet.
        :param temp: Temperature of the system in Kelvins.
        :param c_len: Column length.
        :param u_in: Speed at the inlet.
        :param void_frac: Void fraction (epsilon).
        :param disp: Array containing dispersion coefficient for every component.
        :param kl: Array containing effective mass transport coefficient of every component.
        :param p_he: Helium pressure.
        :param rho_p: Density of the adsorbent.
        """
        # Dimensionless points in time
        self.t_end = t_end * u_in / c_len
        self.dt = dt * u_in / c_len
        self.nt = self.t_end / self.dt

        # Initializing non-dimensionless parameters
        self.p_in = p_in
        self.p_out = p_out

        self.n_points = n_points

        if np.sum(y_in) != 1:
            raise Exception("Sum of mole fractions is not equal to 1")
        self.y_in = y_in

        if temp < 0:
            raise Exception("Temperature cannot be below 0")
        self.temp = temp

        if not 0 <= void_frac <= 1:
            raise Exception("Void fraction is incorrect")
        self.void_frac = void_frac

        self.rho_p = rho_p
        self.p_he = p_he

        # Partial pressures at the inlet
        self.p_partial_in = p_in * y_in
        # Dimensionless mass transfer coefficients
        self.kl = np.asarray(kl) * c_len / u_in
        # Dimensionless dispersion coefficients
        self.disp = np.asarray(disp) / (c_len * u_in)
        # Dimensionless length of the column (always 1)
        self.c_len = 1
        self.dz = self.c_len / (n_points - 1)
        # Dimensionless gradient of the total pressure
        self.dp_dz = (p_out - p_in) / self.c_len
        # Dimensionless inlet velocity (always 1)
        self.v_in = 1

        # The number of components assesses based of the length of y_in array
        self.n_components = len(y_in)

        # p_total is a sum of all partial pressures for each grid point
        self.p_total = np.linspace(p_in, p_out, num=n_points, endpoint=True)



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
        print(self.g_matrix)
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
        print(self.b_vector)
        self.b_vector[0] = -2 * self.params.v_in / self.params.dz

    def calculate_velocity(self, p_partial, q_eq, q_ads):
        """
        Calculates velocity at all grid points.
        :param p_partial: Matrix containing partial pressures of all the components at every grid point.
        Each row represents different grid point.
        :param p_total: Vector containing total pressures at each grid point.
        :param q_eq: Matrix containing equilibrium loadings of each component at each grid point.
        :param q_ads: Matrix containing average loadings in the

        :return: Array containing velocities at each grid point.
        """
        ...

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
        print(m_matrix)
        dp_dt = -np.dot(self.g_matrix, m_matrix) + self.params.disp * np.dot(self.l_matrix, p_partial) - \
            self.params.temp * self.R * ((1 - self.params.void_frac) / self.params.void_frac) * self.params.rho_p *\
            self.params.kl * (q_eq - q_ads) + self.d_matrix
        print("start")
        print(-np.dot(self.g_matrix, m_matrix))
        print(self.params.disp * np.dot(self.l_matrix, p_partial))
        print(self.params.temp * self.R * ((1 - self.params.void_frac) / self.params.void_frac) * self.params.rho_p *\
            self.params.kl * (q_eq - q_ads))
        print(self.d_matrix)
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
        dq_ads_dt = np.transpose(self.params.k_l) * (q_eq - q_ads)
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
        return np.allclose(p_partial_new[0], p_partial_new[p_partial_new.shape[0]-1], 1.e-5)

    def solve(self):
        q_eq = np.zeros((self.params.n_grid_points, self.params.n_components))
        q_ads = np.zeros((self.params.n_grid_points, self.params.n_components))
        p_partial = np.vstack(self.params.p_partial_in, np.zeros((self.params.n_points-1, self.params.n_components)))
        t = 0
        # dpt_dxi = self.calcualte_dpt_dxi
        while (not self.check_equilibrium(p_partial)) or t < self.params.t_end:

            # Calculate new velocity
            v = self.calculate_velocity(p_partial, q_eq, q_ads)

            # Calculate new partial pressures
            dp_dt = self.calculate_dp_dt(v, p_partial, q_eq, q_ads)
            p_partial_new = self.calculate_next_pressure(p_partial, dp_dt)
            if not self.verify_pressures(p_partial_new):
                print("The sum of partial pressures is not equal to 1!")

            # Add something for plotting here

            # Calculate new loadings
            q_eq = ...  # Call the IAST
            dq_ads_dt = self.calculate_dq_ads_dt(q_eq, q_ads)
            q_ads = self.calculate_next_q_ads(q_ads, dq_ads_dt)
            t += self.params.dt

        return p_partial
