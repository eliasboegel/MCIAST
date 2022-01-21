import matplotlib, matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, solver):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.solver = solver
        self.frame = 0
        self.exit_pressure_history = np.empty((self.solver.params.nt, solver.params.n_components - 1))
        self.exit_pressure_history_ms = np.empty((self.solver.params.nt, solver.params.n_components - 1))

        self.fig.suptitle('t = 0.000 s')
        self.ax1.set_xlim(0, 1)
        self.ax1.set_xlabel(r"$\xi$ [-]")
        self.ax1.set_ylabel(r"$Loading$ [mol]")
        self.ax2.set_xlabel(r"t [s]")
        self.ax2.set_ylabel(r"$\frac{y}{y_0}$ [-]", rotation="horizontal")

        # Draw initial state
        loadings = solver.u_0[solver.params.n_points - 1:]
        self.exit_pressure_history[0] = self.solver.u_0[self.solver.params.n_points - 2,
                                        :-1] / self.solver.params.p_partial_in[:-1]
        if self.solver.params.mms is True:
            self.exit_pressure_history_ms[0] = self.solver.MMS.pi_matrix[-1, :-1] / self.solver.params.p_partial_in[:-1]
        for i in range(solver.params.n_components):  # Loop over components
            if self.solver.params.mms is False:
                self.ax1.plot(self.solver.params.xi, loadings[:, i], label=solver.params.component_names[i])
            if self.solver.params.mms is True:
                self.ax1.plot(self.solver.params.xi, loadings[:, i],
                              label=f"Calculated MMS loading, component {i}")
                self.ax1.plot(self.solver.params.xi, self.solver.MMS.q_ads_matrix[:, i],
                              label=f"Real MMS loading, component {i}")
            if i < (solver.params.n_components - 1):
                if self.solver.params.mms is False:
                    self.ax2.plot([0], self.exit_pressure_history[0, i], label=solver.params.component_names[i])
                if self.solver.params.mms is True:
                    self.ax2.plot([0], self.exit_pressure_history[0, i],
                                  label=f"Calculated MMS partial pressure, component {i}")
                    self.ax2.plot([0], self.exit_pressure_history_ms[0, i],
                                  label=f"Real MMS partial pressure, component {i}")
        self.ax1.legend()
        self.ax2.legend()

        plt.ion()
        plt.show(block=False)

    def pause(self, interval):
        """
        Pauses the graph for a give interval.
        :param interval:  The interval to pause for.
        """
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return

    def plot(self, t):
        """Plots the graph depicting adsorbed loadings and breakthrough curves"""
        self.frame += 1

        # Clear figures
        self.ax1.clear()
        self.ax2.clear()

        self.fig.suptitle(f't = {t:.3f} s')
        self.ax1.set_xlabel(r"$\xi$ [-]")
        self.ax1.set_ylabel(r"$Loading$ [$mole/kg$]")
        self.ax2.set_xlabel(r"t [s]")
        self.ax2.set_ylabel(r"$\frac{y}{y_0}$ [-]")

        # Get data
        loadings = self.solver.u_1[self.solver.params.n_points - 1:]
        self.exit_pressure_history[self.frame] = self.solver.u_1[self.solver.params.n_points - 2,
                                                 :-1] / self.solver.params.p_partial_in[:-1]
        if self.solver.params.mms is True:
            self.solver.MMS.update_source_functions(t)
            self.exit_pressure_history_ms[self.frame] = self.solver.MMS.pi_matrix[-1,
                                                        :-1] / self.solver.params.p_partial_in[:-1]

        # Update plots
        for i in range(self.solver.params.n_components):  # Components
            if self.solver.params.mms is False:
                self.ax1.plot(self.solver.params.xi, loadings[:, i], label=self.solver.params.component_names[i])
            if self.solver.params.mms is True:
                self.ax1.plot(self.solver.params.xi, loadings[:, i], label=f"Calculated MMS loading, component {i}")
                self.ax1.plot(self.solver.params.xi, self.solver.MMS.q_ads_matrix[:, i],
                              label=f"Real MMS loading, component {i}")
            if i < (self.solver.params.n_components - 1):
                if self.solver.params.mms is False:
                    self.ax2.plot(np.linspace(0, t, self.frame), self.exit_pressure_history[0:self.frame, i],
                                  label=self.solver.params.component_names[i])
                if self.solver.params.mms is True:
                    self.ax2.plot(np.linspace(0, t, self.frame), self.exit_pressure_history[0:self.frame, i],
                                  label=f"Calculated MMS pressure, component {i}")
                    self.ax2.plot(np.linspace(0, t, self.frame), self.exit_pressure_history_ms[0:self.frame, i],
                                  label=f"Real MMS pressure, component {i}")
        self.ax1.legend()
        self.ax2.legend()

        self.pause(0.000001)
