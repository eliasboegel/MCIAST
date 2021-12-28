import numpy as np

from src.solver import Solver
from src.system_parameters import SysParams


class OrderOfAccuracy:
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

        order_of_accuracy = np.log((error_list[2] - error_list[1]) / (error_list[1] - error_list[0])) / np.log(2)
        return order_of_accuracy, discretization_list


ooa = OrderOfAccuracy(which="Space", n=5, dt=0.001)
print(ooa.analysis()[0])