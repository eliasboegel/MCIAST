import numpy as np

from src.solver import Solver
from src.system_parameters import SysParams


class OrderOfAccuracy:
    def __init__(self, n, r):
        # Initialize parameters
        self.n = n
        self.r = r

    def analysis(self):
        # Create a list of discretizations that will be used to estimate convergence.
        discretization_list = [self.n, self.r * self.n]
        # Create a list to store errors between calculated and manufactured solutions for discretizations above
        error_list = []
        # Create params class that will store steady state (final solution). It is used for both space and time OoA
        # (we want to check convergence termination as well, hence we do not take simply manufactured solution at
        # termination time in the second case)
        ss_params = SysParams()
        for nodes in discretization_list:
            # Set ss_params with tighter discretization for each run
            ss_params.init_params(t_end=10, dt=0.1, y_in=np.asarray([0.25, 0.25, 0.25]), n_points=nodes,
                                  p_in=1, temp=298, c_len=1, u_in=1, void_frac=1, y_fill_gas=0.25,
                                  disp_fill_gas=1, kl_fill_gas=1, disp=[1, 1, 1], kl=[1, 1, 1],
                                  rho_p=1000, time_stepping_method="RK23", dimensionless=True, mms=True,
                                  mms_mode="steady", mms_convergence_factor=1/1000, atol=1e-15)
            ss_solver = Solver(ss_params)
            error_matrix = None

            # Get dp_dt matrix of manufactured solution (all elements should be zero as it is steady state)
            ss_params.MMS.update_source_functions(0)
            dp_dt_manufactured = ss_params.MMS.dpi_dt_matrix
            # Get manufactured solution for adsorbed loading and partial pressures
            q_ads = ss_params.MMS.q_ads_matrix
            p_partial = ss_params.MMS.pi_matrix
            u = np.concatenate((p_partial.flatten("F"), q_ads.flatten("F")), axis=0)
            # Calculate spatial gradient of these quantities (proper discretization should return zero because
            # of steady state)
            du_dt_calc = ss_solver.calculate_dudt(0, u)
            dp_dt_calc = du_dt_calc[0:ss_params.n_components * (nodes - 1)]

            # Calculate error matrix
            error_matrix = np.abs(dp_dt_calc - dp_dt_manufactured.flatten("F"))

            # Find the biggest error
            print("error matrix is:", error_matrix)
            # error_norm = np.sqrt(np.mean(error_matrix.flatten() ** 2, axis=0))
            error_norm = np.amax(error_matrix)
            error_list.append(error_norm)

        print("error list is:", error_list)
        # Calculate order of accuracy based on convergence and discretization
        order_of_accuracy = np.log((error_list[0]) / (error_list[1])) / np.log(self.r)
        return order_of_accuracy, discretization_list, error_list


ooa = OrderOfAccuracy(100, 2)
ooa, dis_list, error_list = ooa.analysis()
print(ooa)