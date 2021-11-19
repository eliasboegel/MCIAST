class Solver:
    # Gas constant
    R = ...

    def __init__(self, temp, h_t, p_in, c_len, void_frac, disp_coeffs, kl, peclet_coeffs, p_he, ro_p, u_in=1):
        """
        Initializes the solver with the parameters that remain constant throughout the calculations
        and the initial conditions.
        :param temp: Temperature.
        :param h_t: Length of one time step.
        :param p_in: Vector containing partial pressures of each component at the intlet.
        :param c_len: Length of the adsorbing column.
        :param void_frac: Void fraction (epsilon).
        :param disp_coeffs: Array containing dispersion coefficient for every component.
        :param kl: Array containing effective mass transport coefficient of every component
        :param peclet_coeffs: Peclet numbers for each component.
        :param p_he: Helium pressure.
        :param ro_p: Density of the adsorbent
        :param u_in: Interstitial velocity at the inlet.
        """
        self.temp = temp
        self.h_t = h_t
        self.p_in = p_in
        self.c_len = c_len
        self.void_frac = void_frac
        self.disp_coeffs = disp_coeffs
        self.peclet_coeffs = peclet_coeffs
        self.kl = kl
        self.ro_p = ro_p
        self.u_in = u_in
        self.p_he = p_he
        self.n_grid_points = ...
        self.n_components = ...
        self.p_total = ...
        self.g_matrix = ...
        self.l_matrix = ...
        self.d_matrix = ...
        self.b_vector = ...
        self.n_components = ...
        self.n_grid_points = ...

    def calcualte_dpt_dxi(self):
        ...

    def calculate_velocity(self, p_partial, dpt_dxi, q_eq, q_ads):
        """
        Calculates velocity at all grid points.
        :param p_partial: Matrix containing partial pressures of all the components at every grid point.
        Each row represents different grid point.
        :param p_total: Vector containing total pressures at each grid point.
        :param dpt_dxi: Value of the gradient of the total pressure (constant).
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
        ...

    def calculate_next_pressure(self, p_partial_old, dp_dt):
        """
        Steps in time to calculate the partial pressures of each component in the next point of time.
        :param p_partial_old: Matrix containing old partial pressures.
        :param dp_dt: Matrix containing time derivatives of partial pressures of each component at each grid point.
        :return: Matrix containing new partial pressures.
        """
        ...

    def verify_pressures(self, p_partial):
        """
        Verify if all partial pressures sum up to 1.
        :param p_partial
        : Matrix containing partial pressures at each grid point
        :return: True if the pressures sum up to 1, false otherwise
        """
        ...

    def calculate_dq_ads_dt(self, q_eq, q_ads):
        """
        Calculates the time derivative of average component loadings in the adsorbent at each point.
        :param q_eq: Array containing equilibrium loadings of each component.
        :param q_ads: Array containing average component loadings in the adsorbent
        :return: Matrix containing time derivatives of average component loadings in the adsorbent at each point.
        """
        ...

    def calculate_next_q_ads(self, q_ads_old, dq_ads_dt):
        """
        Steps in time to calculate the average loadings in the adsorbent of each component in the next point of time.
        :param q_ads_old: Matrix containing old average loadings.
        :param dq_ads_dt: Matrix containing time derivatives of average loadings of each component at each grid point.
        :return: Matrix containing new average loadings.
        """
        ...

    def check_equilibrium(self, p_partial_old, p_partial_new):
        """
        Checks if the components have reached equilibrium and the adsorption process is finished.
        :param p_partial_old: Matrix containing old partial pressures of all components at every point.
        :param p_partial_new: Matrix containing new partial pressures of all components at every point.
        :return: True if the equilibrium is reached, false otherwise.
        """
        if p_partial_new is None:
            return False
        ...

    def solve(self):
        q_eq = ...
        q_ads = ...
        p_partial_old = None
        p_partial_new = self.p_in
        dpt_dxi = self.calcualte_dpt_dxi
        while not self.check_equilibrium(p_partial_old, p_partial_new):

            # Calculate new velocity
            v = self.calculate_velocity(p_partial_new, dpt_dxi, q_eq, q_ads)

            # Calculate new partial pressures
            dp_dt = self.calculate_dp_dt(v, p_partial_new, q_eq, q_ads)
            p_partial_old = p_partial_new
            p_partial_new = self.calculate_next_pressure(p_partial_old, dp_dt)
            if not self.verify_pressures(p_partial_new):
                print("The sum of partial pressures is not equal to 1!")

            # Add something for plotting here

            # Calculate new loadings
            q_eq = ...  # Call the IAST
            dq_ads_dt = self.calculate_dq_ads_dt(q_eq, q_ads)
            q_ads = self.calculate_next_q_ads(q_ads, dq_ads_dt)

        return p_partial_new
