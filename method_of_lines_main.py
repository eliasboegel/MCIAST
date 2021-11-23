#  from method_of_lines_subroutine import mol #not relivent


#!/usr/bin/python3
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from math import exp
#Settings
plot = True
fast_plot = True

# Grid size and snapshots
m = 400
#number of the grid points
snaps= 400#number of time steps forward


# RHS of ODE system from semi-discretization
def deriv(u,t):
    return -cidx*(np.roll(u,1)-np.roll(u,-1))*0.5

# PDE-related constants
c=0.1
dx=1.0/m
cidx=c/dx
dt=0.1

# Initial condition
uinit=np.empty((m))
for i in range(m):
    x=dx*i
    uinit[i]=exp(-20*(x-0.5)**2)

# Define the times for saving snapshots
time=np.linspace(0,dt*snaps,snaps+1)

# Integrate the semi-discretized PDE
u=odeint(deriv,uinit,time);

# Output results
x = np.linspace(0, 1, m)

t = np.linspace(0, dt*snaps+1, snaps+1)

X, T = np.meshgrid(x, t)
Z = u

fig = plt.figure()
ax = plt.axes(projection='3d')
if plot == True:
    if fast_plot == True:
        ax.contour3D(X, T, Z, 50, cmap='binary')
    else:
        ax.plot_surface(X, T, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('T')
ax.set_zlabel('Z');
plt.show()

