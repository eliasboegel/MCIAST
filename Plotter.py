
print('running')
import matplotlib.pyplot as plt
import numpy as np
import time 
from matplotlib.animation import FuncAnimation


def matplot(loading_matrix):
    
    #intialisation
    numb_grid,numb_componets = np.shape(loading_matrix)
    grid = np.linspace(0,1,num=numb_grid)
    
    plt.xlabel("Dimentionless adsorber length")
    plt.ylabel("Loading")
    
    #seperating loadings into their compoents and plotting them 
    for compoent_numb in range(numb_componets):
        compoent = loading_matrix[:,compoent_numb]
        plt.plot(grid,compoent)
        plt.draw
    plt.pause(0.001)
    plt.clf()
    
    
    
        
        
        
       
        
        
    

    #show plot

    #axes.set_xlim(0, 100)
    #axes.set_ylim(-50, +50)

    
    
    
   

#Making test data
for move in range(200):
    move = move/50
    tgrid = 80
    space = np.linspace(0+move,6.28+move,num=tgrid)
    x =  np.sin(space)
    y =  np.cos(space)
    data = np.concatenate((x,y)).reshape((tgrid, 2), order='F')   
    matplot(data)
    







'''

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')


def init():
    #scale
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    #axis names
    plt.xlabel("Dimentionless adsorber length")
    plt.ylabel("Loading")
    return ln,

def update(loading_matrix):
    
    #intialisation
    numb_grid,numb_componets = np.shape(loading_matrix)
    grid = np.linspace(0,1,num=numb_grid)
    
    #seperating loadings into their compoents and plotting them 
    plt.cla()
    for compoent_numb in range(numb_componets):
        compoent = loading_matrix[:,compoent_numb]
        plt.plot(grid,compoent)
    #xdata.append(frame)
    #ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show() 
'''

'''
def plots():
    global vlgaBuffSorted
    cntr()

    result = collections.defaultdict(list)
    for d in vlgaBuffSorted :
        result[d['event']].append(d)

    result_list = result.values()

    f = Figure()
    graph1 = f.add_subplot(211)
    graph2 = f.add_subplot(212,sharex=graph1)

    for item in result_list:
        tL = []
        vgsL = []
        vdsL = []
        isubL = []
        for dict in item:
            tL.append(dict['time'])
            vgsL.append(dict['vgs'])
            vdsL.append(dict['vds'])
            isubL.append(dict['isub'])
        graph1.plot(tL,vdsL,'bo',label='a')
        graph1.plot(tL,vgsL,'rp',label='b')
        graph2.plot(tL,isubL,'b-',label='c')

    plotCanvas = FigureCanvasTkAgg(f, pltFrame)
    toolbar = NavigationToolbar2TkAgg(plotCanvas, pltFrame)
    toolbar.pack(side=BOTTOM)
    plotCanvas.get_tk_widget().pack(side=TOP)
'''
print("done")