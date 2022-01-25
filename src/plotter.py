import matplotlib, matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, sys_params):
        self.params = sys_params

    @staticmethod
    def pause(interval):
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

    def plot(self, t_samples, p_i_evolution, q_ads_evolution):
        """Plots the graph depicting adsorbed loadings and breakthrough curves
        :param t_samples:  Time at which solution is sampled
        :param p_i_evolution: Values of partial pressures at sampled times
        :param q_ads_evolution: Values of adsorbed loadings at sampled times
        """

        # Create figure and axes. ax1 shows loading over the domain at current frame.
        # ax2 shows the evolution from 0 up to current frame of outlet pressure except of fill gas
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)

        # If MMS is used, create am array to store time evolution of ideal solution for comparison with calculated one
        if self.params.use_mms is True:
            exit_pressure_history_ms = np.zeros((t_samples.shape[0], self.params.n_components-1))
        for frame in range(0, t_samples.shape[0]):
            # Get data up to current frame
            t = t_samples[0:frame+1]
            loadings = q_ads_evolution[frame]
            exit_pressure_history = p_i_evolution[0:frame+1]
            if self.params.use_mms is True:
                self.params.MMS.update_source_functions(frame)
                exit_pressure_history_ms[frame] = self.params.MMS.pi_matrix[-1, :-1] / self.params.p_partial_in[:-1]
            # Clear figures
            self.ax1.clear()
            self.ax2.clear()
            # Update plots
            self.fig.suptitle(f't = {t[frame]:.3f} s')
            self.ax1.set_xlabel(r"$\xi$ [-]")
            self.ax1.set_ylabel(r"$Loading$ [$mole/kg$]")
            self.ax2.set_xlabel(r"t [s]")
            self.ax2.set_ylabel(r"$\frac{y}{y_0}$ [-]")

            # Plot the result for each component
            for i in range(self.params.n_components-1):
                if self.params.use_mms is False:
                    self.ax1.plot(self.params.xi, loadings[:, i], label=self.params.component_names[i])
                    self.ax2.plot(t, exit_pressure_history[0:frame+1, i], label=self.params.component_names[i])
                # If MMS is used plot ideal results as well for comparison
                if self.params.use_mms is True:
                    self.ax1.plot(self.params.xi, loadings[:, i], label=f"Calculated MMS loading, component {i}")
                    self.ax1.plot(self.params.xi, self.params.MMS.q_ads_matrix[:, i],
                                  label=f"Real MMS loading, component {i}")
                    self.ax2.plot(t, exit_pressure_history[0:frame+1, i],
                                  label=f"Calculated MMS pressure, component {i}")
                    self.ax2.plot(t, exit_pressure_history_ms[0:frame+1, i],
                                  label=f"Real MMS pressure, component {i}")

            self.ax1.legend()
            self.ax2.legend()
            self.pause(0.001)
            plt.ion()
            plt.show()

