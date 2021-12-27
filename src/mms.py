import numpy as np


class MMS:
    def __init__(self, sys_params):
        # Initialize other classes as attributes
        self.__params = sys_params
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
        self.c = self.__params.mms_conv_factor
        self.t_factor = 0
        self.RT = self.__params.R * self.__params.temp
        self.void_term = self.__params.void_frac_term * self.RT

    def calculate_pi_ms(self, i):
        self.pi = self.__params.p_partial_in[i] + (-1) ** i * (np.sin(np.pi / 2 * self.__params.xi) +
                                           self.b * self.t_factor * np.sin(np.pi * self.__params.xi))

    def calculate_nu_ms(self):
        self.nu = self.__params.v_in - 0.5 * np.sin(np.pi / 2 * self.__params.xi) + \
                  self.b * self.t_factor * np.sin(np.pi * self.__params.xi) ** 2

    def calculate_q_ads_ms(self, tau, i):
        self.q_ads = self.b * self.__params.kl[i] * self.__params.xi * tau * self.t_factor

    def calculate_q_eq_ms(self, tau):
        self.q_eq = self.b * self.__params.xi * (self.t_factor - tau * self.t_factor / self.c) + self.q_ads

    def calculate_dpi_dz_ms(self, i):
        self.dpi_dz = -self.a / (2 * self.__params.n_components) + \
                      (-1) ** i * np.pi * (0.5 * np.cos(np.pi / 2 * self.__params.xi) +
                                           self.b * self.t_factor * np.cos(np.pi * self.__params.xi))

    def calculate_d2pi_dz2_ms(self, i):
        self.d2pi_dz2 = -(-1) ** i * np.pi ** 2 * (0.25 * np.sin(np.pi / 2 * self.__params.xi) +
                                                   self.b * self.t_factor * np.sin(np.pi * self.__params.xi))

    def calculate_dpi_dt_ms(self, i):
        self.dpi_dt = self.b * (-(-1) ** i * self.t_factor * np.sin(np.pi * self.__params.xi)) / self.c

    def calculate_dnu_dz_ms(self):
        self.dnu_dz = np.pi * (-0.25 * np.cos(np.pi / 2 * self.__params.xi) +
                               self.b * 2 * self.t_factor * np.sin(np.pi * self.__params.xi) * np.cos(
                    np.pi * self.__params.xi))

    def calculate_dnupi_dz(self):
        self.dnupi_dz = self.pi * self.dnu_dz + self.nu * self.dpi_dz

    def update_source_functions(self, tau):
        # Update time factor
        self.t_factor = np.e ** (-tau / self.c)
        # Update MS velocity for all components
        self.calculate_nu_ms()
        # Update 1st derivative of MS velocity for all components
        self.calculate_dnu_dz_ms()
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
        self.pi_diffusion_matrix = self.d2pi_dz2_matrix * self.__params.disp_matrix
        self.S_pi = self.dpi_dt_matrix + self.dnupi_dz_matrix - self.pi_diffusion_matrix + \
                    self.void_term * self.__params.kl_matrix * self.delta_q_matrix
        self.S_nu = self.dnu_dz + np.sum((self.void_term * self.__params.kl_matrix *
                                          self.delta_q_matrix - self.pi_diffusion_matrix), axis=1) / self.__params.p_total
