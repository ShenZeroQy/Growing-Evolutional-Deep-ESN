from matplotlib.axis import Axis
#from matplotlib import projections
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from enum import Enum
import os

# align these pram from runner before analying

REP=10
G_num=10
X_add=10
# X_dim=[40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40]
X_dim=[20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
# X_dim=[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]



Predir='./MG2/EVA/SSM1/m'

#for matplot GUI
def inverse(temp, position):
    pass


#def ce

#for  fname


class SingleLineFName(Enum): 
    ASE=0
    AMI=1
    NRMSE_test=2
    NRMSE_train=3
    MAX_corr=4
    Lamda=5
class EVAFName(Enum): 
    EVAASE=0
    EVAAMI=1
    EVANRMSE_test=2
    EVANRMSE_train=3

# class MethodName(Enum): 
#     distance =0
#     pearson_corr =1
#     Spearman_corr=2
#     kendal_corr=3
#     fack=4
#     # Random=4
# class MethodGUIName(Enum): 
#     ED =0
#     PC =1
#     SC =2
#     KC =3
#     RS =4
#     # Random=4    
class MethodName(Enum): 
    distance =0

class MethodGUIName(Enum): 
    ED =0

    Random=1    
#https://zhuanlan.zhihu.com/p/65220518
class PlotColor(Enum): 
    blue=0
    deeppink=1
    firebrick=5
    darkgray=2
    aquamarine=4
    gold=3
    lemonchiffon=105
    beige=6
    bisque=7
    black=8
    blanchedalmond=9

__font_size=20


def AverlizeOVerall(ldir):
    #overall

    outdir=ldir+'aveg/'  
    try:
        os.makedirs(outdir)
        print (outdir+'创建成功')
    except:
        print('Fail:!!!!!!!!!mkdir'+outdir)
    
    for fname in EVAFName:
        Lines=list()
        for i in range(REP):
            indir=ldir+str(i)+'/'
            inname=fname.name+'.txt'
            L=loadtxt(indir+inname,dtype=double, delimiter=',')
            Lines.append(L)
        avegLines=mean(Lines,axis=0)
        stdLines=std(Lines,axis=0)
        savetxt(outdir+'aveg'+inname,avegLines,fmt='%f', delimiter=',')
        savetxt(outdir+'std'+inname,stdLines,fmt='%f', delimiter=',')
        print(outdir+'a-s'+inname+'wrote')

def LayerMerge(ldir,equal=1):
    txtoutdir=ldir+'aveg/'
    for fname in SingleLineFName:
        inname=fname.name+'.txt'
        SDA=list()
        for i in range(REP):    
            figoutdir=ldir+str(i)+'/out/'
            try:
                os.makedirs(figoutdir)
                print (figoutdir+'创建成功')
            except:
                print('Fail:!!!!!!!!!mkdir'+figoutdir) 
                
            
                #layer by layer  
                Lines=list()    
                for l in range(G_num):
                    indir=ldir+str(i)+'/'+'layer'+str(l)+'/'
                    L=loadtxt(indir+inname,dtype=double, delimiter=',')
                    Lines.append(L)
                #all layer done
                if(equal):
                    #To One Lines
                    Lc=Lines[0]
                    for k in range(1,len(Lines)):
                        Lc=concatenate((Lc,Lines[k]),axis=0)
                    SDA.append(Lc)
                #draw fig
                fig=plt.figure(1,figsize=(10,5))
                fig.clear()
                axia=fig.gca()
                xoffset=0
                limmax=L.max()
                limmin=L.min()
                tick=list()
                ticklabel=list()
                tick.append(0)
                ticklabel.append(str(X_dim[0]+X_add))
                ticklayer=['']+ [j+1 for  j in range(G_num)]
                for j in range(G_num):
                    L=Lines[j]
                    xt= [j*X_add+jj for  jj in range(len(L))]
                    tick.append(xt[-1])
                    
                    xoffset=xoffset+X_dim[j]
                    if(j==G_num-1):
                        ticklabel.append(str(xoffset))
                    else:
                        ticklabel.append(str(xoffset)+'\n(+'+str(X_add+X_dim[j])+')')
                    axia.plot(xt,L,label='layer'+str(j))
                    # axia.set_ylim(limmin,limmax)
                    
                    
                axia.set_xticks(tick)
                axia.set_xticklabels(ticklabel)
                axia.set_xlabel('Total neuron number')
                axia.set_ylabel(fname.name)

                secax = axia.secondary_xaxis('top')
                secax.set_xticklabels(ticklayer)
                secax.set_xlabel('Layer number')
                plt.savefig(figoutdir+fname.name+'.eps')
                fig.show()
                plt.close(fig)
def DrawMat(Mat,alphabetx,alphabety,fname):
 
    figure = plt.figure() 
    axes = figure.add_subplot(111) 
    
    # using the matshow() function  
    caxes = axes.matshow(Mat, interpolation ='nearest') 
    figure.colorbar(caxes) 
    
    axes.set_xticklabels(['']+alphabetx) 
    axes.set_yticklabels(['']+alphabety) 
    plt.savefig(fname+'.eps')
    plt.show()

def SDA1():
    Method_SDA_aveg=list()
    Method_SDA_std=list()
    for rj in MethodDirName:
        #assign name
        #rjn=MethodName(int(rj.value))
        rjdn=DataSetEVA+'/'+rj.name+'/'
        #alloc storage
        #analying DAFName(1).name - overfittingfactor
        repeat_SDA=list()
        for rk in range(REP):
            rkdn=rjdn+str(rk)+'/'
            readFname1=rkdn+SingleLineFName(1).name+'.txt'
            readFname2=rkdn+SingleLineFName(2).name+'.txt'
            A1=loadtxt(readFname1,dtype=double, delimiter=',')
            A2=loadtxt(readFname2,dtype=double, delimiter=',')
            A1=(A1-A2)/A2
            savetxt(DAFName(1).name,A1,fmt='%f', delimiter=',')
            repeat_SDA.append(A1)
        aveg_SDA=mean(repeat_SDA,axis=0)
        std_SDA=std(repeat_SDA,axis=0)
        #savefile of aveg result
        savetxt(rjdn+'aveg'+DAFName(1).name+'.txt',aveg_SDA,fmt='%f', delimiter=',')
        savetxt(rjdn+'std'+DAFName(1).name+'.txt',std_SDA,fmt='%f', delimiter=',')
        print(rjdn+'a_s'+DAFName(1).name+'wrote')
        #end saving
        Method_SDA_aveg.append(aveg_SDA)
        Method_SDA_std.append(std_SDA)
    #compare different methods
    fig=plt.figure(1)
    #help for drowing:
    #help for drowing:https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html?highlight=errorbar#matplotlib.axes.Axes.errorbar
    fig.clear()
    x=zeros((aveg_SDA.size,))
    for i in range(aveg_SDA.size):
        x[i]=i
    for rj in MethodName:
        aveg_SDA=Method_SDA_aveg[rj.value]
        std_SDA=Method_SDA_std[rj.value]            
        plt.errorbar(x, aveg_SDA,std_SDA/10, marker='.',ms=0.6,ls="-", lw=0.2,color=PlotColor(rj.value).name, label=rj.name) 
    plt.legend()
    plt.title('Evaluation '+DAFName(1).name+' comparation of different methods')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(inverse))
    plt.savefig(DataSetEVA+'/Evaluation'+DAFName(1).name+' comparation.eps') 
    plt.close(plt.figure(1))


    #CE and div CE
    Method_SDA1_aveg=list()
    Method_SDA1_std=list()
    Method_SDA2_aveg=list()
    Method_SDA2_std=list()
    Method_SDA3_aveg=list()
    Method_SDA3_std=list()
    for rj in MethodDirName:
        #assign name
        #rjn=MethodName(int(rj.value))
        rjdn=DataSetEVA+'/'+rj.name+'/'
        #alloc storage
        #analying CE_nor and div CE
        repeat_SDA1=list()
        repeat_SDA2=list()
        repeat_SDA3=list()
        for rk in range(REP):
            rkdn=rjdn+str(rk)+'/'
            readFname1=rkdn+SingleLineFName(1).name+'.txt'
            readFname2=rkdn+SingleLineFName(2).name+'.txt'

            readFname3=rkdn+SingleLineFName(0).name+'.txt'
            A1=loadtxt(readFname1,dtype=double, delimiter=',')
            A2=loadtxt(readFname2,dtype=double, delimiter=',')
            A3=loadtxt(readFname3,dtype=double, delimiter=',')
            A4=zeros_like(A3)
            for i in range(A3.size):
                #A4[i]=(A3[i]-idealCmef(Node_num-i))/((Node_num-i)**2-idealCmef(Node_num-i))
                A4[i]=A3[i]/(Node_num-i)**2
                A1[i]=1/(A3[i]*A1[i]**alaphy)
                A2[i]=1/(A3[i]*A2[i]**alaphy)
            savetxt(DAFName(2).name+'.txt',A4,fmt='%f', delimiter=',')
            savetxt(DAFName(3).name+'.txt',A1,fmt='%f', delimiter=',')
            savetxt(DAFName(4).name+'.txt',A2,fmt='%f', delimiter=',')
            repeat_SDA1.append(A4)
            repeat_SDA2.append(A1)
            repeat_SDA3.append(A2)
        aveg_SDA=mean(repeat_SDA1,axis=0)
        std_SDA=std(repeat_SDA1,axis=0)
        #savefile of aveg result
        savetxt(rjdn+'aveg'+DAFName(2).name+'.txt',aveg_SDA,fmt='%f', delimiter=',')
        savetxt(rjdn+'std'+DAFName(2).name+'.txt',std_SDA,fmt='%f', delimiter=',')
        print(rjdn+'a_s'+DAFName(2).name+'wrote')
        #end saving
        Method_SDA1_aveg.append(aveg_SDA)
        Method_SDA1_std.append(std_SDA)

        aveg_SDA=mean(repeat_SDA2,axis=0)
        std_SDA=std(repeat_SDA2,axis=0)
        #savefile of aveg result
        savetxt(rjdn+'aveg'+DAFName(3).name+'.txt',aveg_SDA,fmt='%f', delimiter=',')
        savetxt(rjdn+'std'+DAFName(3).name+'.txt',std_SDA,fmt='%f', delimiter=',')
        print(rjdn+'a_s'+DAFName(3).name+'wrote')
        #end saving
        Method_SDA2_aveg.append(aveg_SDA)
        Method_SDA2_std.append(std_SDA)

        aveg_SDA=mean(repeat_SDA3,axis=0)
        std_SDA=std(repeat_SDA3,axis=0)
        #savefile of aveg result
        savetxt(rjdn+'aveg'+DAFName(4).name+'.txt',aveg_SDA,fmt='%f', delimiter=',')
        savetxt(rjdn+'std'+DAFName(4).name+'.txt',std_SDA,fmt='%f', delimiter=',')
        print(rjdn+'a_s'+DAFName(4).name+'wrote')
        #end saving
        Method_SDA3_aveg.append(aveg_SDA)
        Method_SDA3_std.append(std_SDA)
    #compare different methods
    fig=plt.figure(1)
    #help for drowing:
    #help for drowing:https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html?highlight=errorbar#matplotlib.axes.Axes.errorbar
    fig.clear()
    x=zeros((aveg_SDA.size,))
    for i in range(aveg_SDA.size):
        x[i]=i
    for rj in MethodName:
        aveg_SDA=Method_SDA1_aveg[rj.value]
        std_SDA=Method_SDA1_std[rj.value]            
        plt.errorbar(x, aveg_SDA,std_SDA/10, marker='.',ms=0.6,ls="-", lw=0.2,color=PlotColor(rj.value).name, label=rj.name) 
    plt.legend()
    plt.title('Evaluation '+DAFName(2).name+' comparation of different methods')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(inverse))
    plt.savefig(DataSetEVA+'/Evaluation'+DAFName(2).name+' comparation.eps') 
    plt.show()
    plt.close(plt.figure(1))

    fig=plt.figure(1)
    #help for drowing:
    #help for drowing:https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html?highlight=errorbar#matplotlib.axes.Axes.errorbar
    fig.clear()
    x=zeros((aveg_SDA.size,))
    for i in range(aveg_SDA.size):
        x[i]=i
    for rj in MethodName:
        aveg_SDA=Method_SDA2_aveg[rj.value]
        std_SDA=Method_SDA2_std[rj.value]            
        plt.errorbar(x, aveg_SDA,std_SDA/10, marker='.',ms=0.6,ls="-", lw=0.2,color=PlotColor(rj.value).name, label=rj.name) 
    plt.legend()
    plt.title('Evaluation '+DAFName(3).name+' comparation of different methods')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(inverse))
    plt.savefig(DataSetEVA+'/Evaluation'+DAFName(3).name+' comparation.eps') 
    plt.show()
    plt.close(plt.figure(1))

    fig=plt.figure(1)
    #help for drowing:
    #help for drowing:https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html?highlight=errorbar#matplotlib.axes.Axes.errorbar
    fig.clear()
    x=zeros((aveg_SDA.size,))
    for i in range(aveg_SDA.size):
        x[i]=i
    for rj in MethodName:
        aveg_SDA=Method_SDA3_aveg[rj.value]
        std_SDA=Method_SDA3_std[rj.value]            
        plt.errorbar(x, aveg_SDA,std_SDA/10, marker='.',ms=0.6,ls="-", lw=0.2,color=PlotColor(rj.value).name, label=rj.name) 
    plt.legend()
    plt.title('Evaluation '+DAFName(4).name+' comparation of different methods')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(inverse))
    plt.savefig(DataSetEVA+'/Evaluation'+DAFName(4).name+' comparation.eps') 
    plt.show()
    plt.close(plt.figure(1))
def SDA2():
    
    #allocate storage 
    for ri in SingleLineFName:
        #assign name
        rin=ri.name
        rifn=rin+'.txt'
        #alloc storage
        Method_SDA_aveg=list()
        Method_SDA_std=list()
        for rj in MethodDirName:
           
            #assign name
            #rjn=MethodName(int(rj.value))
            rjdn=DataSetEVA+'/'+rj.name+'/'
            #alloc storage
            repeat_SDA=list()
            for rk in range(REP):
                rkdn=rjdn+str(rk)+'/'
                readFname=rkdn+rifn
                print('loading:'+readFname)
                A=loadtxt(readFname,dtype=double, delimiter=',')
                repeat_SDA.append(A)
            aveg_SDA=mean(repeat_SDA,axis=0)
            std_SDA=std(repeat_SDA,axis=0)
            #savefile of aveg result
            savetxt(rjdn+'aveg'+rifn,aveg_SDA,fmt='%f', delimiter=',')
            savetxt(rjdn+'std'+rifn,std_SDA,fmt='%f', delimiter=',')
            print(rjdn+'a_s'+rifn+'wrote')
            #end saving
            Method_SDA_aveg.append(aveg_SDA)
            Method_SDA_std.append(std_SDA)
        #compare different methods
        fig=plt.figure(1)
        #help for drowing:
        #help for drowing:https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html?highlight=errorbar#matplotlib.axes.Axes.errorbar
        fig.clear()
        x=zeros((aveg_SDA.size,))
        for i in range(aveg_SDA.size):
            x[i]=i
        for rj in MethodName:
            aveg_SDA=Method_SDA_aveg[rj.value]
            std_SDA=Method_SDA_std[rj.value]            
            plt.errorbar(x, aveg_SDA,std_SDA, marker='.',ms=0.2,ls="-", lw=0.2,color=PlotColor(rj.value).name, label=rj.name) 
        plt.legend()
        plt.title('Evaluation '+rin+' comparation of different methods')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(inverse))
        plt.savefig(DataSetEVA+'/Evaluation'+rin+' comparation.eps') 
        plt.show()
        plt.close(plt.figure(1))
def SDA3():
    #allocate storage 
    #assign name deallina ASE
    rin= MultiLineFName(0).name
    rifn=rin+'.txt'
    #alloc storage
    Method_SDA_aveg=list()
    Method_SDA_std=list()
    for rj in MethodDirName:

        #assign name
        #rjn=MethodName(int(rj.value))
        rjdn=DataSetEVA+'/'+rj.name+'/'
        #alloc storage
        repeat_SDA=list()
        for rk in range(REP):
            rkdn=rjdn+str(rk)+'/'
            readFname=rkdn+rifn
            print('loading:'+readFname)
            A=loadtxt(readFname,dtype=double, delimiter=',')
            repeat_SDA.append(A)
        aveg_SDA=mean(repeat_SDA,axis=0)
        std_SDA=std(repeat_SDA,axis=0)
        #savefile of aveg result
        savetxt(rjdn+'aveg'+rifn,aveg_SDA,fmt='%f', delimiter=',')
        savetxt(rjdn+'std'+rifn,std_SDA,fmt='%f', delimiter=',')
        print(rjdn+'a_s'+rifn+'wrote')
        #end saving
        Method_SDA_aveg.append(aveg_SDA)
        Method_SDA_std.append(std_SDA)
    #compare different methods
    fig=plt.figure(1,figsize=(4,10))
    #help for drowing:https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html?highlight=errorbar#matplotlib.axes.Axes.errorbar
    fig.clear()
    #generage alabar
    #draw 10%
    alabarx=[str(i+1) for i in range(G_num)]
    alabary=['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%']
    alabary=['0','10','20','30','40','50','60','70','80','90']
    #alabary=['0%','20%','40%','60%','80%']
    # for i in range(G_num):
    #     alabarx.append(str(i+1))
    #alabarx=alabarx.to_array()
    
    for rj in MethodDirName:
        if(rj.value>=4):
            break
        rjdn=DataSetEVA+'/'+rj.name+'/'
        M=loadtxt(rjdn+'aveg'+rifn,dtype=double, delimiter=',').T
        ind=PecPreindex(Node_num).astype(int32)
        GUImat=M[:,ind]
        for i in range(G_num):
            for j in range(GUImat.shape[1]):
                if(GUImat[i,j]==0):
                    GUImat[i,j]=inf
        axes = fig.add_subplot(4,1,rj.value+1)
        #axes.figsize=(1,2)
        im=axes.matshow(GUImat, cmap=plt.cm.Greens)
        if(rj.value==2):
            rm=im
        axes.xaxis.set_major_locator(plt.MaxNLocator(10))
        axes.set_xticklabels(['']+alabary) 
        axes.yaxis.get_major_locator()
        axes.set_ylabel('('+ MethodGUIName(rj.value).name+')      ',rotation=0, x=-0.2,y=0.5,size=12.5)
        axes.set_yticklabels(['']+alabarx) 
        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
    plt.subplots_adjust(bottom=0.05, left=0,right=0.96, top=0.95)
    cax = plt.axes([0.82, 0.05, 0.025, 0.9])
    plt.colorbar(rm ,cax=cax)
    plt.savefig(DataSetEVA+'/'+Me+'_Evaluation'+rin+'.eps') 
    plt.savefig(Outdir+'/'+Me+'_Evaluation'+rin+'.eps') 


    #allocate storage 
    #assign name deallina NodeStack
    rin= MultiLineFName(1).name
    rifn=rin+'.txt'
    #alloc storage
    Method_SDA_aveg=list()
    Method_SDA_std=list()
    for rj in MethodDirName:
        #assign name
        #rjn=MethodName(int(rj.value))
        rjdn=DataSetEVA+'/'+rj.name+'/'
        #alloc storage
        repeat_SDA=list()
        for rk in range(REP):
            rkdn=rjdn+str(rk)+'/'
            readFname=rkdn+rifn
            print('loading:'+readFname)
            A=loadtxt(readFname,dtype=double, delimiter=',')
            repeat_SDA.append(A)
        aveg_SDA=mean(repeat_SDA,axis=0)
        std_SDA=std(repeat_SDA,axis=0)
        #savefile of aveg result
        savetxt(rjdn+'aveg'+rifn,aveg_SDA,fmt='%f', delimiter=',')
        savetxt(rjdn+'std'+rifn,std_SDA,fmt='%f', delimiter=',')
        print(rjdn+'a_s'+rifn+'wrote')
        #end saving
        Method_SDA_aveg.append(aveg_SDA)
        Method_SDA_std.append(std_SDA)
    #compare different methods
    fig=plt.figure(2, figsize=(3.8, 10))
    #help for drowing:https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html?highlight=errorbar#matplotlib.axes.Axes.errorbar
    fig.clear()
    #generage alabar
    #draw 10%
    alabarx=[str(i+1) for i in range(G_num)]
    alabary=['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%']
    alabary=['0','10','20','30','40','50','60','70','80','90']
    #alabary=['0%','20%','40%','60%','80%']
    # for i in range(G_num):
    #     alabarx.append(str(i+1))
    #alabarx=alabarx.to_array()
    
    for rj in MethodDirName:
        if(rj.value>=4):
            break
        rjdn=DataSetEVA+'/'+rj.name+'/'
        M=loadtxt(rjdn+'aveg'+rifn,dtype=double, delimiter=',').T
        ind=PecPreindex(Node_num).astype(int32)
        GUImat=M[:,ind]
        axes = fig.add_subplot(4,1,rj.value+1)
        #axes.figsize=(1,2)
        im=axes.matshow(GUImat,cmap=plt.cm.Reds)
        if(rj.value==2):
            rm=im
        axes.xaxis.set_major_locator(plt.MaxNLocator(10))
        axes.set_xticklabels(['']+alabary) 
        axes.yaxis.get_major_locator()
        axes.set_ylabel('('+MethodGUIName(rj.value).name+')      ',rotation=0, x=-0.2,y=0.5,size=12.5)
        axes.set_yticklabels(['']+alabarx) 
        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
    plt.subplots_adjust(bottom=0.05, left=0,right=0.96, top=0.95)
    cax = plt.axes([0.82, 0.05, 0.025, 0.9])
    plt.colorbar(rm ,cax=cax)
    plt.savefig(DataSetEVA+'/'+Me+'_Evaluation'+rin+'.eps') 
    plt.savefig(Outdir+'/'+Me+'_Evaluation'+rin+'.eps') 
    plt.show()
    plt.close(plt.figure(1))
    plt.close(plt.figure(2))
def SDA3h():
    #allocate storage 
    #assign name deallina ASE
    rin= MultiLineFName(0).name
    rifn=rin+'.txt'
    #alloc storage
    Method_SDA_aveg=list()
    Method_SDA_std=list()
    for rj in MethodDirName:

        #assign name
        #rjn=MethodName(int(rj.value))
        rjdn=DataSetEVA+'/'+rj.name+'/'
        #alloc storage
        repeat_SDA=list()
        for rk in range(REP):
            rkdn=rjdn+str(rk)+'/'
            readFname=rkdn+rifn
            print('loading:'+readFname)
            A=loadtxt(readFname,dtype=double, delimiter=',')
            repeat_SDA.append(A)
        aveg_SDA=mean(repeat_SDA,axis=0)
        std_SDA=std(repeat_SDA,axis=0)
        #savefile of aveg result
        savetxt(rjdn+'aveg'+rifn,aveg_SDA,fmt='%f', delimiter=',')
        savetxt(rjdn+'std'+rifn,std_SDA,fmt='%f', delimiter=',')
        print(rjdn+'a_s'+rifn+'wrote')
        #end saving
        Method_SDA_aveg.append(aveg_SDA)
        Method_SDA_std.append(std_SDA)
    #compare different methods
    fig=plt.figure(1,figsize=(3.5,10))
    #help for drowing:https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html?highlight=errorbar#matplotlib.axes.Axes.errorbar
    fig.clear()
    #generage alabar
    #draw 10%
    alabarx=[str(i+1) for i in range(G_num)]
    alabary=['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%']
    alabary=['0','10','20','30','40','50','60','70','80','90']
    #alabary=['0%','20%','40%','60%','80%']
    # for i in range(G_num):
    #     alabarx.append(str(i+1))
    #alabarx=alabarx.to_array()
    
    for rj in MethodDirName:
        if(rj.value>=4):
            break
        rjdn=DataSetEVA+'/'+rj.name+'/'
        M=loadtxt(rjdn+'aveg'+rifn,dtype=double, delimiter=',').T
        ind=PecPreindex(Node_num).astype(int32)
        GUImat=M[:,ind]
        for i in range(G_num):
            for j in range(GUImat.shape[1]):
                if(GUImat[i,j]==0):
                    GUImat[i,j]=inf
        axes = fig.add_subplot(5,1,rj.value+1)
        #axes.figsize=(1,2)
        im=axes.matshow(GUImat, cmap=plt.cm.Greens)
        if(rj.value==2):
            rm=im
        axes.xaxis.set_major_locator(plt.MaxNLocator(10))
        axes.set_xticklabels(['']+alabary) 
        axes.yaxis.get_major_locator()
        axes.set_ylabel('('+ MethodGUIName(rj.value).name+')      ',rotation=0, x=-0.2,y=0.5,size=12.5)
        axes.set_yticklabels(['']+alabarx) 
        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
    plt.subplots_adjust(bottom=0.05, left=0,right=0.96, top=0.95)
    cax = plt.axes([0.82, 0.05, 0.025, 0.9])
    plt.colorbar(rm ,cax=cax)
    plt.savefig(DataSetEVA+'/'+Me+'_Evaluation'+rin+'.eps') 
    plt.savefig(Outdir+'/'+Me+'_Evaluation'+rin+'.eps') 


    #allocate storage 
    #assign name deallina NodeStack
    rin= MultiLineFName(1).name
    rifn=rin+'.txt'
    #alloc storage
    Method_SDA_aveg=list()
    Method_SDA_std=list()
    for rj in MethodDirName:
        #assign name
        #rjn=MethodName(int(rj.value))
        rjdn=DataSetEVA+'/'+rj.name+'/'
        #alloc storage
        repeat_SDA=list()
        for rk in range(REP):
            rkdn=rjdn+str(rk)+'/'
            readFname=rkdn+rifn
            print('loading:'+readFname)
            A=loadtxt(readFname,dtype=double, delimiter=',')
            repeat_SDA.append(A)
        aveg_SDA=mean(repeat_SDA,axis=0)
        std_SDA=std(repeat_SDA,axis=0)
        #savefile of aveg result
        savetxt(rjdn+'aveg'+rifn,aveg_SDA,fmt='%f', delimiter=',')
        savetxt(rjdn+'std'+rifn,std_SDA,fmt='%f', delimiter=',')
        print(rjdn+'a_s'+rifn+'wrote')
        #end saving
        Method_SDA_aveg.append(aveg_SDA)
        Method_SDA_std.append(std_SDA)
    #compare different methods
    fig=plt.figure(2, figsize=(3.5, 10))
    #help for drowing:https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html?highlight=errorbar#matplotlib.axes.Axes.errorbar
    fig.clear()
    #generage alabar
    #draw 10%
    alabarx=[str(i+1) for i in range(G_num)]
    alabary=['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%']
    alabary=['0','10','20','30','40','50','60','70','80','90']
    #alabary=['0%','20%','40%','60%','80%']
    # for i in range(G_num):
    #     alabarx.append(str(i+1))
    #alabarx=alabarx.to_array()
    
    for rj in MethodDirName:
        rjdn=DataSetEVA+'/'+rj.name+'/'
        M=loadtxt(rjdn+'aveg'+rifn,dtype=double, delimiter=',').T
        ind=PecPreindex(Node_num).astype(int32)
        GUImat=M[:,ind]
        axes = fig.add_subplot(5,1,rj.value+1)
        #axes.figsize=(1,2)
        im=axes.matshow(GUImat,cmap=plt.cm.Reds)
        if(rj.value==2):
            rm=im
        axes.xaxis.set_major_locator(plt.MaxNLocator(10))
        axes.set_xticklabels(['']+alabary) 
        axes.yaxis.get_major_locator()
        axes.set_ylabel('('+MethodGUIName(rj.value).name+')      ',rotation=0, x=-0.2,y=0.5,size=12.5)
        axes.set_yticklabels(['']+alabarx) 
        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
    plt.subplots_adjust(bottom=0.05, left=0,right=0.96, top=0.95)
    cax = plt.axes([0.82, 0.05, 0.025, 0.9])
    plt.colorbar(rm ,cax=cax)
    plt.savefig(DataSetEVA+'/'+Me+'_Evaluation'+rin+'.eps') 
    plt.savefig(Outdir+'/'+Me+'_Evaluation'+rin+'.eps') 
    plt.show()
    plt.close(plt.figure(1))
    plt.close(plt.figure(2))

def Analysis_eva():
    SDA3()
    #SDA1()
    SDA2()
    return
def Load_rep():
    #only aveg
    eva_ASE_mean=zeros((10,1))
    eva_nrmse_train=zeros((10,1))
    eva_nrmse_test=zeros((10,1))
    for p in range(10):
        dir=DataSetREC+'/'+str(p)+'/'
        print('loading:'+dir)
        test=loadtxt(dir+SingleLineFName(1).name+'.txt',dtype=double, delimiter=',').T
        eva_nrmse_test[p]=mean(test)
        train=loadtxt(dir+SingleLineFName(2).name+'.txt',dtype=double, delimiter=',').T
        eva_nrmse_train[p]=mean(train)
        ase=loadtxt(dir+SingleLineFName(3).name+'.txt',dtype=double, delimiter=',').T
        eva_ASE_mean[p]=mean(ase)
    return  eva_nrmse_test,eva_nrmse_train ,eva_ASE_mean
def Load_eva(meind:int):
    eva_ASE_mean=zeros((10,1)) 
    eva_nrmse_train=zeros((10,1))
    eva_nrmse_test=zeros((10,1))

    dir=DataSetEVA+'/'+MethodDirName(meind).name+'/'        
    print('loading:'+dir)
    test=loadtxt(dir+'aveg'+SingleLineFName(1).name+'.txt',dtype=double, delimiter=',').T  
    train=loadtxt(dir+'aveg'+SingleLineFName(2).name+'.txt',dtype=double, delimiter=',').T    
    ase=loadtxt(dir+'aveg'+SingleLineFName(3).name+'.txt',dtype=double, delimiter=',').T 
    for p in range(10):
        eva_nrmse_test[p]=test[int(p*Node_num/10)]
        eva_nrmse_train[p]=train[int(p*Node_num/10)]
        eva_ASE_mean[p]=ase[int(p*Node_num/10)]
    return  eva_nrmse_test,eva_nrmse_train ,eva_ASE_mean
def photo():
    pind=PecPreindex(Node_num)
    return
def CmpNRMSE_ASE():
    rtr,rte,ra=Load_rep()
    etr0,ete0,ea0=Load_eva(0)
    etr1,ete1,ea1=Load_eva(1)
    etr2,ete2,ea2=Load_eva(2)
    etr3,ete3,ea3=Load_eva(3)
    etr4,ete4,ea4=Load_eva(4)
    #pare NrMse te
    fig=plt.figure(1,  figsize=(12, 8)) 
    axia=plt.gca()
    axia.plot(range(10),rtr,color='red', linewidth=1.0, linestyle='-', label='Unpuned deepESN',marker='^',markersize=10)
    axia.plot(range(10),etr0,color='blue', linewidth=1.0, linestyle='-', label='NSIPA-ED',marker='*',markersize=10)
    axia.plot(range(10),etr1,color='orange', linewidth=1.0, linestyle='-', label='NSIPA-PC',marker='h',markersize=10)
    axia.plot(range(10),etr2,color='gray', linewidth=1.0, linestyle='-', label='NSIPA-SC',marker='X',markersize=10)
    axia.plot(range(10),etr3,color='pink', linewidth=1.0, linestyle='-', label='NSIPA-KC',marker='P',markersize=10)
    axia.plot(range(10),etr4,color='indigo', linewidth=1.0, linestyle='-', label='NIPA',marker='s',markersize=10)
    #set font
    fs=20
    #axia.set_title('$NRMSE_{test}$ comparation with M='+str(Node_num),fontsize= fs)
    axia.xaxis.set_major_locator(plt.MaxNLocator(10))
    axia.set_xticklabels([' ' ,'0%','10%','20%','30%','40%','50%','60%','70%','80%','90%'])
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    plt.savefig(Outdir+'/'+Me+'_Compare_NRMSE_test.eps')
    fig.show()

    #pare NrMse tr
    fig=plt.figure(2,figsize=(12, 8)) 
    axia=plt.gca()
    axia.plot(range(10),rte,color='red', linewidth=1.0, linestyle='-', label='Unpuned deepESN',marker='^',markersize=10)
    axia.plot(range(10),ete0,color='blue', linewidth=1.0, linestyle='-', label='NSIPA-ED',marker='*',markersize=10)
    axia.plot(range(10),ete1,color='orange', linewidth=1.0, linestyle='-', label='NSIPA-PC',marker='h',markersize=10)
    axia.plot(range(10),ete2,color='gray', linewidth=1.0, linestyle='-', label='NSIPA-SC',marker='X',markersize=10)
    axia.plot(range(10),ete3,color='pink', linewidth=1.0, linestyle='-', label='NSIPA-KC',marker='P',markersize=10)
    axia.plot(range(10),ete4,color='indigo', linewidth=1.0, linestyle='-', label='NIPA',marker='s',markersize=10)


    #axia.set_title('$NRMSE_{train}$ comparation with M='+str(Node_num),fontsize= fs)
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.xaxis.set_major_locator(plt.MaxNLocator(10))
    axia.set_xticklabels([' ' ,'0%','10%','20%','30%','40%','50%','60%','70%','80%','90%'])
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    plt.savefig(Outdir+'/'+Me+'_Compare_NRMSE_train.eps')
    fig.show()

    #pare ASE

    fig=plt.figure(3,figsize=(12, 8)) 
    axia=plt.gca()
    axia.plot(range(10),ra,color='red', linewidth=1.0, linestyle='-', label='Unpuned deepESN',marker='^',markersize=10)
    axia.plot(range(10),ea0,color='blue', linewidth=1.0, linestyle='-', label='NSIPA-ED',marker='*',markersize=10)
    axia.plot(range(10),ea1,color='orange', linewidth=1.0, linestyle='-', label='NSIPA-PC',marker='h',markersize=10)
    axia.plot(range(10),ea2,color='gray', linewidth=1.0, linestyle='-', label='NSIPA-SC',marker='X',markersize=10)
    axia.plot(range(10),ea3,color='pink', linewidth=1.0, linestyle='-', label='NSIPA-KC',marker='P',markersize=10)
    axia.plot(range(10),ea4,color='indigo', linewidth=1.0, linestyle='-', label='NIPA',marker='s',markersize=10)

    #axia.set_title('ASE comparation with M='+str(Node_num),fontsize= fs)
    axia.xaxis.set_major_locator(plt.MaxNLocator(10))
    axia.set_xticklabels([' ' ,'0%','10%','20%','30%','40%','50%','60%','70%','80%','90%'])
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    plt.savefig(Outdir+'/'+Me+'_Compare_ASE.eps')
    plt.show()
    ens=1
def CmpNRMSE_BSE():
    rtr,rte,ra=Load_rep()
    etr0,ete0,ea0=Load_eva(0)
    etr1,ete1,ea1=Load_eva(1)
    etr2,ete2,ea2=Load_eva(2)
    etr3,ete3,ea3=Load_eva(3)
    etr4,ete4,ea4=Load_eva(4)
    #pare NrMse te
    fig=plt.figure(1,  figsize=(15, 9)) 
    axia=plt.gca()
    axia.plot(range(10),rtr,color='red', linewidth=1.0, linestyle='-', label='$EN$',marker='^',markersize=10)
    axia.plot(range(10),etr0,color='blue', linewidth=1.0, linestyle='-', label='$ES$',marker='P',markersize=10)
    axia.plot(range(10),etr1,color='orange', linewidth=1.0, linestyle='-', label='$CS_1$',marker='h',markersize=10)
    axia.plot(range(10),etr2,color='gray', linewidth=1.0, linestyle='-', label='$CS_2$',marker='X',markersize=10)
    axia.plot(range(10),etr3,color='pink', linewidth=1.0, linestyle='-', label='$CS_3$',marker='s',markersize=10)
    axia.plot(range(10),etr4,color='indigo', linewidth=1.0, linestyle='-', label='$RS$',marker='*',markersize=10)
    axia.axhline(0.0258*1.1,color='black',linestyle='--')
    plt.annotate(r"$+10\%\mu$",fontsize=20, xy=(0,0.0258*1.1), xytext=(0.5, 0.026*1.2),
             arrowprops=dict(facecolor="black", shrink=0.01, width=0.8))

    #set font
    fs=20
    axia.set_title('$NRMSE_{test}$ comparation with M='+str(Node_num),fontsize= fs)
    axia.xaxis.set_major_locator(plt.MaxNLocator(10))
    axia.set_xticklabels([' ' ,'0%','10%','20%','30%','40%','50%','60%','70%','80%','90%'])
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    plt.savefig(Outdir+'/NRMSE_test.eps')
    fig.show()

    #pare NrMse tr
    fig=plt.figure(2,figsize=(15, 9)) 
    axia=plt.gca()
    axia.plot(range(10),rte,color='red', linewidth=1.0, linestyle='-', label='DYC',marker='^',markersize=10)
    axia.plot(range(10),ete0,color='blue', linewidth=1.0, linestyle='-', label='EVA_distance',marker='P',markersize=10)
    axia.plot(range(10),ete1,color='orange', linewidth=1.0, linestyle='-', label='EVA_corr_Pearson',marker='h',markersize=10)
    axia.plot(range(10),ete2,color='gray', linewidth=1.0, linestyle='-', label='EVA_corr_Spearman',marker='X',markersize=10)
    axia.plot(range(10),ete3,color='pink', linewidth=1.0, linestyle='-', label='EVA_corr_Kendall',marker='s',markersize=10)
    axia.plot(range(10),ete4,color='indigo', linewidth=1.0, linestyle='-', label='EVA_fack',marker='*',markersize=10)


    axia.set_title('$NRMSE_{train}$ comparation with M='+str(Node_num),fontsize= fs)
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.xaxis.set_major_locator(plt.MaxNLocator(10))
    axia.set_xticklabels([' ' ,'0%','10%','20%','30%','40%','50%','60%','70%','80%','90%'])
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    plt.savefig(Outdir+'/NRMSE_train.eps')
    fig.show()

    #pare ASE

    fig=plt.figure(3,figsize=(15, 9)) 
    axia=plt.gca()
    axia.plot(range(10),ra,color='red', linewidth=1.0, linestyle='-', label='DYC',marker='^',markersize=10)
    axia.plot(range(10),ea0,color='blue', linewidth=1.0, linestyle='-', label='EVA_distance',marker='P',markersize=10)
    axia.plot(range(10),ea1,color='orange', linewidth=1.0, linestyle='-', label='EVA_corr_Pearson',marker='h',markersize=10)
    axia.plot(range(10),ea2,color='gray', linewidth=1.0, linestyle='-', label='EVA_corr_Spearman',marker='X',markersize=10)
    axia.plot(range(10),ea3,color='pink', linewidth=1.0, linestyle='-', label='EVA_corr_Kendall',marker='s',markersize=10)
    axia.plot(range(10),ea4,color='indigo', linewidth=1.0, linestyle='-', label='EVA_fack',marker='*',markersize=10)

    axia.set_title('ASE comparation with M='+str(Node_num),fontsize= fs)
    axia.xaxis.set_major_locator(plt.MaxNLocator(10))
    axia.set_xticklabels([' ' ,'0%','10%','20%','30%','40%','50%','60%','70%','80%','90%'])
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    plt.savefig(Outdir+'/ASE.eps')
    fig.show()
    ens=1
if __name__ == '__main__':
    for m in range(2):
        Lredir=Predir+str(m)+'/'
        AverlizeOVerall(Lredir)
        LayerMerge(Lredir,equal=1)
    end=1
   