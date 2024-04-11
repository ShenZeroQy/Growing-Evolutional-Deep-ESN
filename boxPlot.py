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
Me='SSM'
Predir='./'+Me+'/16P10P'
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
def plott():       
    fs=15
    
    fig=plt.figure(1,  figsize=(12, 6)) 
    axia=plt.gca()
    axia.errorbar(xran,rte,yerr=stdrte,xerr=None,color='red',linewidth=1, linestyle='dashed', label='Deep ESN',marker='X',markersize=10,capsize=4)
    # axia.plot(rte,color='red', linewidth=1.0, linestyle='-', label='Deep ESN',marker='^',markersize=10)
    axia.errorbar(xran,ete,yerr=stdete,xerr=None,color='blue', linewidth=1, linestyle='dashed', label='GE Deep ESN',marker='P',markersize=10,capsize=4)
    # axia.plot(ete,color='blue', linewidth=1.0, linestyle='-', label='GE Deep ESN',marker='P',markersize=10)    
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.set_ylabel('NRMSE_test',size=fs)
    axia.set_xlabel('Layer number',size=fs)
    
    alabarx=[str(j+1) for j in range(G_num)]
    plt.xticks(range(G_num),labels=alabarx)
   
    plt.savefig(repdir+'/'+str(m)+'_Compare_NRMSE_test.eps')
    fig.show()    
    fig.clear()
    fig=plt.figure(2,  figsize=(12, 6)) 
    axia=plt.gca()
    axia.errorbar(xran,rtr,yerr=stdrtr,xerr=None,color='red',linewidth=1.0, linestyle='dashed', label='Deep ESN',marker='X',markersize=10,capsize=4)
    axia.errorbar(xran,etr,yerr=stdetr,xerr=None,color='blue', linewidth=1.0, linestyle='dashed', label='GE Deep ESN',marker='P',markersize=10,capsize=4)
    # axia.plot(rtr,color='red', linewidth=1.0, linestyle='-', label='Deep ESN',marker='X',markersize=10)
    # axia.plot(etr,color='blue', linewidth=1.0, linestyle='-', label='GE Deep ESN',marker='P',markersize=10)    
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.set_ylabel('NRMSE_train',size=fs)
    axia.set_xlabel('Layer number',size=fs)
    
    alabarx=[str(j+1) for j in range(G_num)]
    plt.xticks(range(G_num),labels=alabarx)
    plt.savefig(repdir+'/'+str(m)+'_Compare_NRMSE_train.eps')
    fig.show()   
    fig.clear()    
    return

if __name__ == '__main__':
    repdir=Predir+'5/rep/'
    boxGroup=[]
    boxSizet=[]
    for i in range(3):
        repErr,sizet=BestRep(repdir,rank=i)
        boxGroup.append(repErr)
        boxSizet.append(sizet+1)
        
    # boxlabels=['Deep ESN('+str(boxSizet[i])+')\n'+r'$\varepsilon=0\%$' for i in range(len(boxSizet)) ] 
    boxlabels=['Deep ESN\nLayer size:'+str(boxSizet[i]) for i in range(len(boxSizet)) ] 
    
    
    
    evadir=Predir+'5/m3/'
    evaErr5=BestEva(evadir)
    evadir=Predir+'10/m3/'
    evaErr10=BestEva(evadir)
    boxGroup.append( evaErr5)
    boxGroup.append( evaErr10)
    boxlabels.append('GE Deep ESN\n'+ r'$\varepsilon=33\%$')
    
    boxlabels.append('GE Deep ESN\n'+ r'$\varepsilon=50\%$')
    fig=plt.figure(1,figsize=(7,4) )

    axia=plt.gca()  
    axia.boxplot(boxGroup,labels=boxlabels,
                 showmeans=True,meanline=True,sym='+')

   
    axia.set_ylabel('NRMSE_test')

    fig.tight_layout()
    plt.savefig('./'+Me+'/boxMinNrmseTest.eps')
    plt.show()
    end=1
    
