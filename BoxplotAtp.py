from matplotlib.axis import Axis
#from matplotlib import projections
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.widgets import MultiCursor
from enum import Enum
import os

G_num=16
REP=20
Me='ATP'
Predir='./'+Me+'/'
expsize=11
def BestEva(evadir):
    fle=loadtxt(evadir+'/grow/NRMSE_test.txt')
    return fle
def BestRep(repdir,rank=0):
    #load rte,tr
    # xran=[i for i in range(G_num)]
    rte=zeros(G_num,)
    stdrte=zeros(G_num,)

    for i in range (G_num):
        fle=loadtxt(repdir+str(i)+'/REPNRMSE_test.txt')
        rte[i]=mean(fle)
        stdrte[i]=std(fle)
    idx = argsort(rte)
    ind=idx[rank]
    # bestnrmsetest=min(rte)
    # bestnrmsetest=median(rte)
    # bestnrmsetest=rte[expsize]
    
    
    # ind=argwhere(rte==bestnrmsetest)
    fle=loadtxt(repdir+str(ind)+'/REPNRMSE_test.txt')  
    return fle,ind 

if __name__ == '__main__':
    repdir=Predir+'REP/3/'
    evadir=Predir+'EVA/m3/'
    boxGroupA=[]
    boxGroupE=[]
    fle=loadtxt(evadir+'/MSEtest.txt')
    flr=loadtxt(evadir+'/MSEtrain.txt')
    boxGroupA.append(fle[:,0])
    boxGroupE.append(fle[:,1])
    # boxGroupA.append(flr[:,0])
    # boxGroupE.append(flr[:,1])
    
    fle=loadtxt(repdir+'/MSEtest.txt')
    flr=loadtxt(repdir+'/MSEtrain.txt')
    boxGroupA.append(fle[:,0])
    boxGroupE.append(fle[:,1])
    # boxGroupA.append(flr[:,0])
    # boxGroupE.append(flr[:,1])

        
    # boxlabels=['Deep ESN('+str(boxSizet[i])+')\n'+r'$\varepsilon=0\%$' for i in range(len(boxSizet)) ] 
    fig,(axia,axib) = plt.subplots(2,sharex=True) 
    axia.boxplot(boxGroupA,
                 showmeans=True,meanline=True,sym='+')

    axib.boxplot(boxGroupE,
                 showmeans=True,meanline=True,sym='+')
    axia.set_ylabel('NRMSE_test')

    fig.tight_layout()
    # plt.savefig('./'+Me+'/boxMSETest.eps')
    plt.show()
    end=1
    
