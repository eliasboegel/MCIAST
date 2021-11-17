class Solver:
    # Gas constant
    R = ...

    def __init__(self, temp,  h_t, p_in, c_len, void_frac, disp_coeffs, kl, peclet_coeffs, phe, u_in=1):
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
        :param phe: Helium pressure.
        :param u_in: Interstitial velocity at the inlet.
        """
        temp = temp
        h_t = h_t
        p_in = p_in
        c_len = c_len
        void_frac = void_frac
        disp_coeffs = disp_coeffs
        peclet_coeffs = peclet_coeffs
        kl = kl
        u_in = u_in
        phe = phe
        g_matrix = ...
        l_matrix = ...
        d_matrix = ...

    def calcualte_dpt_dxi(self):
        ...

    def calculate_velocity(self, p_total, dpt_dxi, q_eq, q_ads, pressures):
        """
        Calculates velocity at all grid points.
        :param p_total: Vector containing total pressures at each grid point.
        :param dpt_dxi: Value of the gradient of the total pressure (constant).
        :param q_eq: Matrix containing equilibrium loadings of each component at each grid point.
        :param q_ads: Matrix containing average loadings in the
        :param pressures: Matrix containing partial pressures of all the components at every grid point.
        Each row represents different grid point.
        :return: Array containing velocities at each grid point.
        """
        ...

    def calculate_dp_dt(self, velocities, pressures, q_eq, q_ads, ro_p):
        """
        Calculates the time derivatives of partial pressures of each component for each grid point.
        Each row represents one grid point. Each column represents one component.
        :param velocities: Array containing velocities at each grid point.
        :param pressures: Matrix containing partial pressures of every component at each grid point.
        :param q_eq: Matrix containing equilibrium loadings of every component at each grid point.
        :param q_ads: Matrix containing average loadings in the adsorbent of every component at each grid point.
        :param ro_p: Density of the adsorbent (?)
        :return: Matrix containing the time derivatives of partial pressures of each component at each grid point.
        """
        ...

    def calculate_next_p(self, p_old, dp_dt):
        """
        Steps in time to calculate the partial pressures of each component in the next point of time.
        :param p_old: Matrix containing old partial pressures.
        :param dp_dt: Matrix containing time derivatives of partial pressures of each component at each grid point.
        :return: Matrix containing new partial pressures.
        """
        ...

    def verify_pressures(self, p):
        """
        Verify if all partial pressures sum up to 1.
        :param p: Array containing partial pressures at each grid point
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

    def solve(self):
        ...
