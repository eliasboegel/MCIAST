import os
from mms import MMS
import numpy as np
import scipy.sparse as sp
import iast


class SysParams:
    def __init__(self):
        self.spatial_discretization_method = 0
        self.u_0 = 0
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
        self.use_mms = 0
        self.mms_mode = 0
        self.mms_conv_factor = 0
        self.ms_pt_distribution = 0
        self.outlet_boundary_type = 0
        self.void_frac_term = 0
        self.atol = 0
        self.time_stepping_scheme = 0
        self.isotherms = 0
        self.R = 0
        self.t_samples = 0
        self.MMS = 0
        # Initializing matrices
        self.kl_matrix = 0
        self.disp_matrix = 0
        self.g_matrix = 0
        self.f_matrix = 0
        self.l_matrix = 0
        self.d_matrix = 0
        self.b_v_vector = 0
        self.e_vector = 0
        self.xi = 0

    def init_params(self, y_in, n_points, p_in, temp, c_len, u_in, void_frac, disp, kl, rho_p,
                    t_end, dt, y_fill_gas, disp_fill_gas, kl_fill_gas, time_stepping_method, atol, dimensionless=True,
                    mms=False, mms_mode="transient", mms_convergence_factor=1000,
                    spatial_discretization_method="central"):

        """
        Initializes the solver with the parameters that remain constant throughout the calculations
        and the initial conditions. Depending on the dimensionless parameter, the variables might turned into the
        dimensionless equivalents. The presence of helium is implicit. It means that it is always present no matter
        what parameters are passed. Its pressure is equal to the pressure of all components at the inlet.
        Therefore, the number of components is always len(y_in)+1.


        :param spatial_discretization_method: Sets whether to use upwind or central method
        :param atol: Absolute error for linear solvers and time stepping schemes
        :param t_end: Final time point.
        :param dt: Length of one time step.
        :param dimensionless: Boolean that specifies whether dimensionless numbers are used.
        :param time_stepping_method: String that specifies the time of stepping methods.
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
        :param kl_fill_gas: mass transfer coefficient of helium, should be 0 by default
        :param disp_fill_gas: dispersion coefficient of helium
        :param y_fill_gas: mole fraction of helium at the inlet, should be 0 by default
        :param mms: Choose if dynamic code testing is switched on.
        :param mms_mode: Choose if MMS is to be used to steady state or transient simulation.
        :param mms_convergence_factor: Choose how quickly MMS is supposed to reach steady state.
        """

        p_out = p_in  # This could be changed to user input if discretization and vectorization is ever fixed.
        ms_pt_distribution = "constant"  # Same as above

        self.R = 8.314

        if dimensionless:
            # Dimensionless quantities
            self.t_end = t_end * u_in / c_len
            self.dt = dt * u_in / c_len
            self.nt = int(self.t_end / self.dt)
            self.y_in = np.asarray(y_in)
            self.kl = np.asarray(kl) * c_len / u_in
            self.disp = np.asarray(disp) / (c_len * u_in)

        else:
            # Quantities with dimensions
            self.t_end = t_end
            self.dt = dt
            self.nt = int(self.t_end / self.dt)
            self.y_in = np.asarray(y_in)
            self.kl = np.asarray(kl)
            self.disp = np.asarray(disp)

        # Appending the parameters of helium to the component arrays
        self.kl = np.append(self.kl, kl_fill_gas)
        self.y_in = np.append(self.y_in, y_fill_gas)
        self.disp = np.append(self.disp, disp_fill_gas)

        # The number of components assessed based on the length of y_in array (so that it includes helium)
        self.n_components = self.y_in.shape[0]
        if self.use_mms is True and (self.n_components % 2) != 0:
            raise Warning("Number of components for MMS must be even ")

        # Specify inlet and outlet pressures and number of points
        self.p_in = p_in
        self.p_out = p_out
        self.n_points = n_points

        # Save chosen discretization method
        self.spatial_discretization_method = spatial_discretization_method
        if self.spatial_discretization_method != "upwind" and self.spatial_discretization_method != "central":
            raise Warning("spatial_discretization_method must be either upwind or central")

        # Throw an Exception if mole fractions do not equal to 1
        if np.sum(self.y_in) != 1.0:
            raise Exception("Sum of mole fractions is not equal to 1")
        if temp < 0:
            raise Exception("Temperature cannot be below 0")
        self.temp = temp
        if not 0 <= void_frac <= 1:
            raise Exception("Void fraction is incorrect")

        # Specify more constants
        self.void_frac = void_frac
        self.rho_p = rho_p
        # Used to calculate velocity
        self.void_frac_term = ((1 - self.void_frac) / self.void_frac) * self.rho_p

        # Dimensionless length of the column (always 1)
        self.c_len = 1
        self.dz = self.c_len / (n_points - 1)

        # Dimensionless inlet velocity (always 1)
        self.v_in = 1

        # Parameters for running dynamic code verification using MMS
        self.use_mms = mms
        if self.use_mms is True:
            self.mms_mode = mms_mode
            self.mms_conv_factor = mms_convergence_factor
            self.ms_pt_distribution = ms_pt_distribution
            if dimensionless is False:
                raise Warning("mms must be initialized with dimensionless set to True")
        if type(self.use_mms) != bool:
            raise Warning("mms must be either True or False")

        # Create array holding nodes coordinates (without node 0)
        self.xi = np.linspace(0, self.c_len, self.n_points)[1:]
        # Set total pressures and its gradient for MMS
        if self.use_mms is True:
            # If total pressure is constant over xi, set it to 1 and its gradient to 0
            if self.ms_pt_distribution == "constant":
                self.p_total = np.full(self.n_points - 1, 1)
                self.dp_dz = 0
            # If total pressure is not constant over xi, set it and its gradient
            elif self.ms_pt_distribution == "linear":
                self.p_total = 1 - self.xi / 2
                self.dp_dz = - 1 / 2
            else:
                self.p_total = 1
                raise Warning("ms_pt_distribution needs to be either constant or linear!")
            # Set MMS partial pressures at the inlet to total pressure divided over components
            self.p_partial_in = np.full(self.n_components, 1 / self.n_components)

        # Set inlet pressure, total pressure and its gradient (in case MMS is not initialized)
        else:
            self.p_partial_in = self.y_in * p_in
            self.p_total = np.linspace(p_in, p_out, n_points)[1:]
            self.dp_dz = (p_out - p_in) / self.c_len

        # Parameters for outlet boundary condition
        if self.p_in == self.p_out or (self.ms_pt_distribution == "constant" and self.use_mms is True):
            self.outlet_boundary_type = "Neumann"
        elif self.p_in != self.p_out or (self.ms_pt_distribution == "linear" and self.use_mms is True):
            self.outlet_boundary_type = "Numerical"
        elif self.outlet_boundary_type != "Neumann" and self.outlet_boundary_type != "Numerical":
            raise Warning("Outlet boundary condition needs to be either Neumann or Numerical")

        # Determine the magnitude of error to be set for convergence and for linear solver
        self.time_stepping_scheme = time_stepping_method
        self.atol = atol
        self.t_samples = np.arange(0.0, self.t_end, self.dt)

        # Initialize matrices with parameters set
        self.initialize_matrices()

        # Initialize MMS
        if self.use_mms is True:
            self.MMS = MMS(self)

        # Create initial conditions, partial pressures for helium at the beginning are equal to total pressure
        q_ads_initial = np.zeros((self.n_points - 1, self.n_components))
        p_partial_initial = np.zeros((self.n_points - 1, self.n_components))
        if self.use_mms is False:
            p_partial_initial[:, -1] = self.p_total
        elif self.use_mms is True:
            self.MMS.update_source_functions(0)
            p_partial_initial = self.MMS.pi_matrix
        self.u_0 = np.concatenate((p_partial_initial.flatten("F"), q_ads_initial.flatten("F")), axis=0)

        # IAST stuff
        # dirpath = os.path.abspath(os.path.dirname(__file__))
        # self.component_names, self.isotherms = iast.fit([dirpath + "/test_data/n2.csv", dirpath + "/test_data/co2.csv"])
        self.component_names = np.array(["CO2", "N2", "He"])
        self.isotherms = np.array([
            [1, 3.317e-4],  # CO2
            [0.3, 1e-5]  # N2
        ])

    def initialize_matrices(self):
        """
        Initializes all the constant matrices used in the solver using the already defined system parameters.
        """
        # Those matrices will be used in the solver, their internal structure is fully explained in the report

        # Dimensionless mass transfer coefficients matrix
        self.kl_matrix = np.broadcast_to(self.kl, (self.n_points - 1, self.n_components))

        # Dimensionless dispersion coefficients matrix
        self.disp_matrix = np.broadcast_to(self.disp, (self.n_points - 1, self.n_components))

        # Gradient matrix
        if self.spatial_discretization_method == "upwind":
            self.g_matrix = np.diag(np.full(self.n_points - 1, 3.0), 0) + np.diag(
                np.full(self.n_points - 2, -4.0), -1) + np.diag(np.full(self.n_points - 3, 1.0), -2)
            self.g_matrix[0, 0] = 0.0
            self.g_matrix[0, 1] = 1.0
            self.g_matrix = self.g_matrix / (2.0 * self.dz)
            self.g_matrix = sp.csr_matrix(self.g_matrix)

            # Create matrices and vectors with boundary terms
            self.d_matrix = np.zeros((self.n_points - 1, self.n_components))
            first_row = self.p_partial_in * ((self.v_in / (2 * self.dz)) + (self.disp / (self.dz ** 2)))
            second_row = -self.p_partial_in * (self.v_in / (2 * self.dz))
            self.d_matrix[0] = first_row
            self.d_matrix[1] = second_row

            self.b_v_vector = np.zeros(self.n_points - 1)
            self.b_v_vector[0] = - self.v_in / (2 * self.dz)
            self.b_v_vector[1] = self.v_in / (2 * self.dz)

        elif self.spatial_discretization_method == "central":
            self.g_matrix = np.diag(np.full(self.n_points - 2, -1.0), -1) + np.diag(
                np.full(self.n_points - 2, 1.0), 1)
            self.g_matrix[-1, -3] = 1.0
            self.g_matrix[-1, -2] = -4.0
            self.g_matrix[-1, -1] = 3.0
            self.g_matrix = self.g_matrix / (2.0 * self.dz)
            self.g_matrix = sp.csr_matrix(self.g_matrix)

            # Create matrices and vectors with boundary terms
            self.d_matrix = np.zeros((self.n_points - 1, self.n_components))
            first_row = self.p_partial_in * ((self.v_in / (2 * self.dz)) + (self.disp / (self.dz ** 2)))
            self.d_matrix[0] = first_row

            self.b_v_vector = np.zeros(self.n_points - 1)
            self.b_v_vector[0] = - self.v_in / (2 * self.dz)

        # F matrix for calculating velocity
        self.f_matrix = np.diag(self.dp_dz / self.p_total)
        self.f_matrix = sp.csr_matrix(self.f_matrix)

        # Laplacian operator matrix
        self.l_matrix = np.diag(np.full(self.n_points - 2, 1.0), -1) + np.diag(
            np.full(self.n_points - 2, 1.0), 1) + np.diag(np.full(self.n_points - 1, -2.0), 0)
        if self.outlet_boundary_type == "Neumann":
            self.l_matrix[-1, -2] = 2.0
        elif self.outlet_boundary_type == "Numerical":
            self.l_matrix[-1, -4] = -1.0
            self.l_matrix[-1, -3] = 4.0
            self.l_matrix[-1, -2] = -5.0
            self.l_matrix[-1, -1] = 2.0
        self.l_matrix /= self.dz ** 2
        self.l_matrix = sp.csr_matrix(self.l_matrix)

        # Create matrix for inlet boundary condition for laplacian operator on partial pressures
        self.e_vector = np.zeros(self.n_points - 1)
        self.e_vector[0] = - np.sum(self.p_partial_in * self.disp) / (self.dz ** 2)
