
print('running')
import matplotlib.pyplot as plt
import numpy as np
import time 
from matplotlib.animation import FuncAnimation


def animate(loading_matrix):
    #intialisation
    numb_grid,numb_componets = np.shape(loading_matrix)
    grid = np.linspace(0,1,num=numb_grid)
    
    #seperating loadings into their compoents and plotting them 
    plt.cla()
    for compoent_numb in range(numb_componets):
        compoent = loading_matrix[:,compoent_numb]
        plt.plot(grid,compoent)

    #show plot
    '''
    axes.set_xlim(0, 100)
    axes.set_ylim(-50, +50)
    '''
    plt.xlabel("Dimentionless adsorber length")
    plt.ylabel("Loading")
    
def initplot(loading_matrix): 
    ani = FuncAnimation(plt.gcf(), animate(loading_matrix),interval=50)
    plt.polt
   

#Making test data
for move in range(4):
    tgrid = 80
    space = np.linspace(0+move,6.28+move,num=tgrid)
    x =  np.sin(space)
    y =  np.cos(space)
    data = np.concatenate((x,y)).reshape((tgrid, 2), order='F')   
    time.sleep(0.01)
    initplot(data)


print("done")