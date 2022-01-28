# MCIAST
A numerical code on breakthrough curve modelling for multicomponent gas mixtures based on Ideal Adsorption Theory.

### Running the code
Before running the code, make sure that you have installed all the packages contained in the requirements.txt. To see an example of how our system works, simply run the solver.py file with the Python3 interpreter. 
To adjust the system with your own parameters, follow the example seen in the fuction run_simulation inside of the solver.py file. 
Firstly, the object of the SysParams class should be created and initialized with the parameters characterizing the simulated system.
Then, a solver should be created using those parameters. The simulation is started by calling the solve() method by the solver object. 
This function returns arrays filled with the simulation data which, in turn, can be fed into the plotter to graph the breakthrough curves and adsorbent loadings.  