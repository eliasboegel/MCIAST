import numpy as np

from src.solver import Solver
from src.system_parameters import SysParams


class OrderOfAccuracy:
    def __init__(self, which, n, dt, r):
        # Initialize parameters
        self.n = n
        self.dt = dt
        self.type = which
        self.r = r
        if self.type != "Space" and self.type != "Time":
            raise Warning("which of OoM must be either Space or Time")

    def analysis(self):
        # Create a list of discretizations that will be used to estimate convergence.
        discretization_list = [(self.dt, self.n), (self.dt / self.r, self.r * self.n),
                               (self.dt / (self.r**2), self.r**2 * self.n)]
        # Create a list to store errors between calculated and manufactured solutions for discretizations above
        error_list = []
        # Create params class that will store steady state (final solution). It is used for both space and time OoA
        # (we want to check convergence termination as well, hence we do not take simply manufactured solution at
        # termination time in the second case)
        ss_params = SysParams()
        for (dt, nodes) in discretization_list:
            # Set ss_params with tighter discretization for each run
            ss_params.init_params(t_end=0.001, dt=dt, y_in=np.asarray([0.25, 0.25, 0.25]), n_points=nodes,
                                  p_in=1, temp=298, c_len=1, u_in=1, void_frac=1, y_fill_gas=0.25,
                                  disp_fill_gas=1, kl_fill_gas=1, disp=[1, 1, 1], kl=[1, 1, 1],
                                  rho_p=1000, p_out=0.5, time_stepping_method="CN", dimensionless=True, mms=True,
                                  ms_pt_distribution="linear", mms_mode="steady", mms_convergence_factor=1/100000)
            ss_solver = Solver(ss_params)
            error_matrix = None
            if self.type == "Space":
                # Get dp_dt matrix of manufactured solution (all elements should be zero as it is steady state)
                ss_solver.MMS.update_source_functions(0)
                dp_dt_manufactured = ss_solver.MMS.dpi_dt_matrix
                # Get manufactured solution for adsorbed loading and partial pressures
                q_ads = ss_solver.MMS.q_ads_matrix
                p_partial = ss_solver.MMS.pi_matrix
                u = np.concatenate((p_partial, q_ads), axis=0)
                # Calculate spatial gradient of these quantities (proper discretization should return zero because
                # of steady state)
                du_dt_calc = ss_solver.calculate_dudt(u, 0)
                dp_dt_calc = du_dt_calc[0:nodes - 1]

                # Calculate error matrix
                error_matrix = np.abs(dp_dt_calc - dp_dt_manufactured)

            elif self.type == "Time":
                # Create time parameters and solver
                t_params = SysParams()
                t_params.init_params(t_end=0.001, dt=dt, y_in=np.asarray([0.25, 0.25, 0.25]), n_points=nodes,
                                     p_in=1, temp=298, c_len=1, u_in=1, void_frac=1, y_fill_gas=0.25,
                                     disp_fill_gas=1, kl_fill_gas=1, disp=[1, 1, 1], kl=[1, 1, 1],
                                     rho_p=1000, p_out=0.5, time_stepping_method="CN", dimensionless=True, mms=True,
                                     ms_pt_distribution="linear", mms_mode="transient", mms_convergence_factor=1/100000)
                t_solver = Solver(t_params)

                # Get the calculated solution
                t_calc, p_i_calc, q_ads_calc = t_solver.solve()

                # Get the manufactured solution
                ss_solver.MMS.update_source_functions(0)
                p_i_manufactured = ss_solver.MMS.pi_matrix

                # Calculate error matrix
                error_matrix = np.abs(p_i_calc[-1] - p_i_manufactured[-1])

            # Use RMS error over nodes and then over components (to make the largest error have the biggest weight)
            print("error matrix is:", error_matrix)
            # error_norm = np.sqrt(np.mean(error_matrix.flatten() ** 2, axis=0))
            error_norm = np.amax(error_matrix)
            error_list.append(error_norm)

        print("error list is:", error_list)
        # Calculate order of accuracy based on convergence and discretization
        order_of_accuracy = np.log((error_list[2] - error_list[1]) / (error_list[1] - error_list[0])) / np.log(1/self.r)
        return order_of_accuracy, discretization_list
