
print('running')

import matplotlib.pyplot as plt
import numpy as np

def plot(loading_matrix):
    
    #intialisation
    numb_grid,numb_componets = np.shape(loading_matrix)
    grid = np.linspace(0,1,num=numb_grid)
    
    #Plot paramiters
    plt.xlabel("Dimentionless adsorber length")
    plt.ylabel("Loading")
    
    #seperating loadings into their compoents and plotting them 
    for compoent_numb in range(numb_componets):
        compoent = loading_matrix[:,compoent_numb]
        plt.plot(grid,compoent)
        plt.draw
        
    #Clear the plot 
    plt.pause(0.0001)  #For some reason it does not work without this pause 
    plt.clf()       

#Making test data
for move in range(100):
    move = move/50
    tgrid = 50
    space = np.linspace(0+move,6.28+move,num=tgrid)
    x =  np.sin(space)
    y =  np.cos(space)
    data = np.concatenate((x,y)).reshape((tgrid, 2), order='F')   
    plot(data)
    
print("done")