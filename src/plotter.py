import matplotlib, matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, solver):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.solver = solver
        self.frame = 0
        self.exit_pressure_history = np.empty((self.solver.params.nt, solver.params.component_names.shape[0] - 1))

        self.fig.suptitle('t = 0.000 s')
        self.ax1.set_xlim(0, 1)
        self.ax1.set_xlabel(r"$\xi$ [-]")
        self.ax1.set_ylabel(r"$Loading$ [mol]")
        self.ax2.set_xlabel(r"t [s]")
        self.ax2.set_ylabel(r"$\frac{y}{y_0}$ [-]", rotation="horizontal")

        # Draw initial state
        loadings = solver.u_0[solver.params.n_points - 1:]
        self.exit_pressure_history[0] = self.solver.u_0[self.solver.params.n_points - 2,:-1] / self.solver.params.p_partial_in[:-1]
        for i in range(solver.params.component_names.shape[0]): # Loop over components
            self.ax1.plot(np.linspace(0, 1, solver.params.n_points - 1), loadings[:,i], label=solver.params.component_names[i])
            if i < (solver.params.component_names.shape[0] - 1): self.ax2.plot([0], self.exit_pressure_history[0,i])
        self.ax1.legend()

        plt.ion()
        plt.show(block=False)

    def pause(self, interval):
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
        self.exit_pressure_history[self.frame] = self.solver.u_1[self.solver.params.n_points - 2,:-1] / self.solver.params.p_partial_in[:-1]

        # Update plots
        for i in range(self.solver.params.component_names.shape[0]): # Components
            self.ax1.plot(np.linspace(1/self.solver.params.n_points, 1, self.solver.params.n_points - 1), loadings[:,i], label=self.solver.params.component_names[i])
            if i < (self.solver.params.component_names.shape[0] - 1): self.ax2.plot(np.linspace(0, t, self.frame), self.exit_pressure_history[0:self.frame, i])
        self.ax1.legend()

        self.pause(0.01)