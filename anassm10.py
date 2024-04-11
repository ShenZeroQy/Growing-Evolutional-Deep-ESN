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

# align these pram from runner before analying

REP=25
G_num=16
X_add=10
# X_dim=[40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40]
# X_dim=[20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
X_dim=[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]


pusai=1e-8
Predir='./SSM/16P10P10/'


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

# def LayerMerge(ldir,equal=1):
#     txtoutdir=ldir+'aveg/'
#     for fname in SingleLineFName:
#         inname=fname.name+'.txt'
#         SDA=list()
#         for i in range(REP):    
#             figoutdir=ldir+str(i)+'/out/'
#             try:
#                 os.makedirs(figoutdir)
#                 print (figoutdir+'创建成功')
#             except:
#                 print('Fail:!!!!!!!!!mkdir'+figoutdir) 
                
            
#                 #layer by layer  
#                 Lines=list()    
#                 for l in range(G_num):
#                     indir=ldir+str(i)+'/'+'layer'+str(l)+'/'
#                     L=loadtxt(indir+inname,dtype=double, delimiter=',')
#                     Lines.append(L)
#                 #all layer done
#                 if(equal):
#                     #To One Lines
#                     Lc=Lines[0]
#                     for k in range(1,len(Lines)):
#                         Lc=concatenate((Lc,Lines[k]),axis=0)
#                     SDA.append(Lc)
#                 #draw fig
#                 fig=plt.figure(1,figsize=(10,5))
#                 fig.clear()
#                 axia=fig.gca()
#                 xoffset=0
#                 limmax=L.max()
#                 limmin=L.min()
#                 tick=list()
#                 ticklabel=list()
#                 tick.append(0)
#                 ticklabel.append(str(X_dim[0]+X_add))
#                 ticklayer=['']+ [j+1 for  j in range(G_num)]
#                 for j in range(G_num):
#                     L=Lines[j]
#                     xt= [j*X_add+jj for  jj in range(len(L))]
#                     tick.append(xt[-1])
                    
#                     xoffset=xoffset+X_dim[j]
#                     if(j==G_num-1):
#                         ticklabel.append(str(xoffset))
#                     else:
#                         ticklabel.append(str(xoffset)+'\n(+'+str(X_add+X_dim[j])+')')
#                     axia.plot(xt,L,label='layer'+str(j))
#                     # axia.set_ylim(limmin,limmax)
                    
                    
#                 axia.set_xticks(tick)
#                 axia.set_xticklabels(ticklabel)
#                 axia.set_xlabel('Total neuron number')
#                 axia.set_ylabel(fname.name)

#                 secax = axia.secondary_xaxis('top')
#                 secax.set_xticklabels(ticklayer)
#                 secax.set_xlabel('Layer number')
#                 plt.savefig(figoutdir+fname.name+'.eps')
#                 fig.show()
#                 plt.close(fig)
       
def mirr(x):
    return x;   
def mirrinv(x):
    return x;              
def LayerMerge_aveg(ldir,equal=1):
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
                ticklayer= ['         '+str(j+1) for  j in range(G_num)]+['']
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

                secax = axia.secondary_xaxis('top',functions=(mirr,mirrinv)) 
                # movetick=[tick[j]+G_num/2 for j in range(G_num+1)]    
                # secax.set_xticks(movetick)
                secax.set_xticks(tick)
                secax.set_xticklabels(ticklayer)
                # secax.set_xticks(tick)
                secax.set_xlabel('Layer number')

                plt.grid(axis='x')
                plt.savefig(figoutdir+fname.name+'.eps')
                fig.show()
                plt.close(fig)
        SDAaveg=mean(SDA,axis=0)
        savetxt(txtoutdir+'U'+fname.name+'.txt',SDAaveg)
        
def GUILayerMerge_aveg(ldir,):
    
    fs = 15
    txtoutdir=ldir+'aveg/'
    for i in range(REP): 
        rdir= ldir+str(i)+'/'           
        #creat a fig
        fig, (ax1,axC, ax2,ax3) = plt.subplots(4, sharex=True,gridspec_kw={'height_ratios': [1,1,2,2]})    
        # fig.figure(figsize=(60,80),dpi=150)
        fig.set_figheight(8)
        fig.set_figwidth(12)
        plt.rcParams['ytick.labelsize']=fs
        plt.rcParams['xtick.labelsize']=fs
        
    #####
        inname='MAX_corr.txt'
        #layer by layer  
        Lines=list()    
        for l in range(G_num):
            indir=ldir+str(i)+'/'+'layer'+str(l)+'/'
            L=loadtxt(indir+inname,dtype=double, delimiter=',')
            Lines.append(L)
            #all layer done

        #To One Lines
        Lc=Lines[0]
        for k in range(1,len(Lines)):
            Lc=concatenate((Lc,Lines[k]),axis=0)
        #draw fig
        axia=ax1
        xoffset=0
        tick=list()
        ticklabel=list()
        tick.append(0)
        ticklabel.append(str(X_dim[0]+X_add))
        ticklayer= ['          '+str(j+1) for  j in range(G_num)]+['']
        for j in range(G_num):
            L=Lines[j]
            xt= [j*X_add+jj for  jj in range(len(L))]
            tick.append(xt[-1])
                    
            xoffset=xoffset+X_dim[j]
            if(j==G_num-1):
                ticklabel.append(str(xoffset))
            else:
                ticklabel.append(str(xoffset)+'\n(+'+str(X_add+X_dim[j])+')')
            axia.plot(xt,L,label='layer'+str(j),color='blue')
                    # axia.set_ylim(limmin,limmax)

        
        
    #####1end
        axia.set_xticks(tick)
        # axia.set_xticklabels(ticklabel)
        # axia.set_xlabel('Total neuron number')
        axia.set_ylabel('MS',fontsize =  fs)
        axia.grid(axis='x')
        axia=ax1
        secax = axia.secondary_xaxis('top',functions=(mirr,mirrinv)) 
        # movetick=[tick[j]+G_num/2 for j in range(G_num+1)]    
        # secax.set_xticks(movetick)
        secax.set_xticks(tick)
        secax.set_xticklabels(ticklayer,fontsize =  fs)
        # secax.set_xticks(tick)
        secax.set_xlabel('Layer number',fontsize =  fs)
    #####
    ##external read C
        EMS=loadtxt(rdir+'_endcritation.txt')
        C=zeros((G_num))
        C[0]=2*pusai
        for k in range(1,G_num):
            C[k]=EMS[k-1]-EMS[k]
        axia=axC
    ##compute C done
    #find index
        Cm=0
        for k in range(G_num):
            if(C[k]<pusai):
                Cm=k
                break;
    ##
        tickmove=zeros((G_num))
        for k in range(G_num):
            tickmove[k]=tick[k]+int(X_add/2)
        # z_ax.grid(axis='y',)
        axia.scatter(tickmove[0:Cm],C[0:Cm],color='blue')
        axia.scatter(tickmove[Cm],C[Cm],color='red',label='End growing')
        axia.axhline(pusai,color='gray',label=r'$\eta$')
        axia.legend(loc=7,fontsize =  fs)#right
        axia.grid(axis='x')
        axia.set_ylabel('C',fontsize =  fs)
        ##
    
    #####
    #####
        inname='NRMSE_train.txt'
        #layer by layer  
        Lines=list()    
        for l in range(G_num):
            indir=ldir+str(i)+'/'+'layer'+str(l)+'/'
            L=loadtxt(indir+inname,dtype=double, delimiter=',')
            Lines.append(L)
            #all layer done

        #To One Lines
        Lc=Lines[0]
        for k in range(1,len(Lines)):
            Lc=concatenate((Lc,Lines[k]),axis=0)
        #draw fig
        axia=ax2
        xoffset=0
        tick=list()
        ticklabel=list()
        tick.append(0)
        ticklabel.append(str(X_dim[0]+X_add))
        ticklayer= ['      '+str(j+1) for  j in range(G_num)]+['']
        for j in range(G_num):
            L=Lines[j]
            xt= [j*X_add+jj for  jj in range(len(L))]
            tick.append(xt[-1])
                    
            xoffset=xoffset+X_dim[j]
            if(j==G_num-1):
                ticklabel.append(str(xoffset))
            else:
                ticklabel.append(str(xoffset)+'\n(+'+str(X_add+X_dim[j])+')')
            axia.plot(xt,L,label='layer'+str(j),color='blue')
                    # axia.set_ylim(limmin,limmax)

    #####2end
        axia.grid(axis='x')
        axia.set_ylabel('NRMSE_train',fontsize =  fs)
    #####
        inname='NRMSE_test.txt'
        #layer by layer  
        Lines=list()    
        for l in range(G_num):
            indir=ldir+str(i)+'/'+'layer'+str(l)+'/'
            L=loadtxt(indir+inname,dtype=double, delimiter=',')
            Lines.append(L)
            #all layer done

        #To One Lines
        Lc=Lines[0]
        for k in range(1,len(Lines)):
            Lc=concatenate((Lc,Lines[k]),axis=0)
        #draw fig
        axia=ax3
        xoffset=0
        tick=list()
        ticklabel=list()
        tick.append(0)
        ticklabel.append(str(X_dim[0]+X_add))
        ticklayer= ['         '+str(j+1) for  j in range(G_num)]+['']
        for j in range(G_num):
            L=Lines[j]
            xt= [j*X_add+jj for  jj in range(len(L))]
            tick.append(xt[-1])
                    
            xoffset=xoffset+X_dim[j]
            if(j==G_num-1):
                ticklabel.append(str(xoffset))
            else:
                ticklabel.append(str(xoffset)+'\n(+'+str(X_add+X_dim[j])+')')
            axia.plot(xt,L,label='layer'+str(j),color='blue')
                    # axia.set_ylim(limmin,limmax)

    #####
        axia=ax3
        # axia.set_xticks(tick)
       
        axia.set_xticklabels(ticklabel,fontsize =  fs)
        axia.set_ylabel('NRMSE_test',fontsize =  fs)
        axia.set_xlabel('Total neuron number',fontsize =  fs)
        axia.grid(axis='x')
        # axia.set_ylabel(fname.name)

        # secax = axia.secondary_xaxis('top',functions=(mirr,mirrinv)) 
        # movetick=[tick[j]+G_num/2 for j in range(G_num+1)]    
        # secax.set_xticks(movetick)
        # secax.set_xticks(tick)
        # secax.set_xticklabels(ticklayer)
        # secax.set_xticks(tick)
        # secax.set_xlabel('Layer number')

        plt.tight_layout() 
        fig.show()
        plt.savefig(rdir+'GrowEvo.eps')
       
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


def photo():
    pind=PecPreindex(Node_num)
    return
def anagrowlayer(evadir,fname,bound=1e-4):
    mintest=zeros((REP,))
    Layerg=zeros((REP,))
    for i in range(REP):
        rdir=evadir+str(i)+'/layer'
        growfac=zeros((G_num,))
        growbelif=zeros((G_num,))
        for g in range(G_num):
            lfdir=rdir+str(g)+'/'
            # print('loading'+lfdir+fname)
            fle=loadtxt(lfdir+fname)
            growfac[g]=mean(fle)
            growbelif[g]=std(fle)
            # growfac[g]=fle[-1]
        
        savetxt(evadir+str(i)+'/_endcritation.txt',growfac) 
        gold=growfac[0]
        get=G_num-1
        for g in range(1,G_num):
            gnew=growfac[g]
            loss=gold-gnew
            # if(loss < bound and growbelif[g]<0.01 and growbelif[g-1]<0.015):
            if(loss < bound and growbelif[g]<0.05 and growbelif[g-1]<0.05 ):
                
                get=g
                break
            gold=gnew
        endl=get
        # print('g='+str(get))
        Layerg[i]=get+1
        exnrmse=loadtxt(rdir+str(endl)+'/'+'NRMSE_test.txt')
        mintest[i]=exnrmse[-1]
    try:
        os.makedirs(evadir+'grow/')
    except:
        print('Fail:!!!!!!!!!mkdir'+evadir+'grow/')
    savetxt(evadir+'grow/'+'NRMSE_test.txt',mintest) 
    savetxt(evadir+'grow/'+'layergrow.txt',Layerg,fmt='%d') 
    meanmin=mean(mintest)

    return meanmin

def ComPareRepAndEva(repdir,evadir,m):
    #load rte,tr
    xran=[i for i in range(G_num)]
    rte=zeros(G_num,)
    stdrte=zeros(G_num,)
    rtr=zeros(G_num,)
    stdrtr=zeros(G_num,)
    for i in range (G_num):
        fle=loadtxt(repdir+str(i)+'/REPNRMSE_test.txt')
        rte[i]=mean(fle)
        stdrte[i]=std(fle)
        fle=loadtxt(repdir+str(i)+'/REPNRMSE_train.txt')
        rtr[i]=mean(fle)
        stdrtr[i]=std(fle)
    #load ete,tr
    ete=zeros(G_num,)
    etr=zeros(G_num,)

    stdete=zeros(G_num,)
    stdetr=zeros(G_num,)

    fle=loadtxt(evadir+'aveg'+'/avegEVANRMSE_test.txt')
    for i in range (G_num):
        ete[i]=fle[2*i+1]
    fle=loadtxt(evadir+'aveg'+'/avegEVANRMSE_train.txt')
    for i in range (G_num):
        etr[i]=fle[2*i+1] 
    # std
    fle=loadtxt(evadir+'aveg'+'/stdEVANRMSE_test.txt')
    for i in range (G_num):
        stdete[i]=fle[2*i+1]
    fle=loadtxt(evadir+'aveg'+'/stdEVANRMSE_train.txt')
    for i in range (G_num):
        stdetr[i]=fle[2*i+1] 
        
    fs=25
    
    fig=plt.figure(1,  figsize=(12, 6)) 
    axia=plt.gca()
    axia.errorbar(xran,rte,yerr=stdrte,xerr=None,color='red',linewidth=1, linestyle='dashed', label='Deep ESN',marker='X',markersize=15,capsize=4)
    # axia.plot(rte,color='red', linewidth=1.0, linestyle='-', label='Deep ESN',marker='^',markersize=10)
    axia.errorbar(xran,ete,yerr=stdete,xerr=None,color='blue', linewidth=1, linestyle='dashed', label='GE Deep ESN',marker='P',markersize=15,capsize=4)
    # axia.plot(ete,color='blue', linewidth=1.0, linestyle='-', label='GE Deep ESN',marker='P',markersize=10)    
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.set_ylabel('NRMSE_test',size=fs)
    axia.set_xlabel('Layer number',size=fs)
    
    alabarx=[str(j+1) for j in range(G_num)]
    plt.xticks(range(G_num),labels=alabarx)
   
    plt.tight_layout()
    plt.savefig(repdir+'/'+str(m)+'_Compare_NRMSE_test.eps')
    fig.show()    
    fig.clear()
    fig=plt.figure(2,  figsize=(12, 6)) 
    axia=plt.gca()
    axia.errorbar(xran,rtr,yerr=stdrtr,xerr=None,color='red',linewidth=1.0, linestyle='dashed', label='Deep ESN',marker='X',markersize=15,capsize=4)
    axia.errorbar(xran,etr,yerr=stdetr,xerr=None,color='blue', linewidth=1.0, linestyle='dashed', label='GE Deep ESN',marker='P',markersize=15,capsize=4)
    # axia.plot(rtr,color='red', linewidth=1.0, linestyle='-', label='Deep ESN',marker='X',markersize=10)
    # axia.plot(etr,color='blue', linewidth=1.0, linestyle='-', label='GE Deep ESN',marker='P',markersize=10)    
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    axia.set_ylabel('NRMSE_train',size=fs)
    axia.set_xlabel('Layer number',size=fs)
    
    alabarx=[str(j+1) for j in range(G_num)]
    plt.xticks(range(G_num),labels=alabarx)
    
    plt.tight_layout()
    plt.savefig(repdir+'/'+str(m)+'_Compare_NRMSE_train.eps')
    fig.show()   
    fig.clear()    
    return
def ComPareRepAndEvaASE(repdir,evadir,m):
    #load rte,tr
    xran=[i for i in range(G_num)]
    rte=zeros(G_num,)
    stdrte=zeros(G_num,)
    # rtr=zeros(G_num,)
    # stdrtr=zeros(G_num,)
    for i in range (G_num):
        fle=loadtxt(repdir+str(i)+'/REPASE.txt')
        rte[i]=mean(fle)
        stdrte[i]=std(fle)
        # fle=loadtxt(repdir+str(i)+'/REPNRMSE_train.txt')
        # rtr[i]=mean(fle)
        # stdrtr[i]=std(fle)
    #load ete,tr
    ete=zeros(G_num,)
    etr=zeros(G_num,)

    stdete=zeros(G_num,)
    stdetr=zeros(G_num,)

    fle=loadtxt(evadir+'aveg'+'/avegEVAASE.txt')
    for i in range (G_num):
        ete[i]=fle[2*i+1]
    # fle=loadtxt(evadir+'aveg'+'/avegEVANRMSE_train.txt')
    # for i in range (G_num):
    #     etr[i]=fle[2*i+1] 
    # std
    fle=loadtxt(evadir+'aveg'+'/stdEVAASE.txt')
    for i in range (G_num):
        stdete[i]=fle[2*i+1]
    # fle=loadtxt(evadir+'aveg'+'/stdEVANRMSE_train.txt')
    # for i in range (G_num):
    #     stdetr[i]=fle[2*i+1] 
    fs=20
    
    fig=plt.figure(1,  figsize=(12, 6)) 
    axia=plt.gca()
    axia.errorbar(xran,rte,yerr=stdrte,xerr=None,color='red',linewidth=1, linestyle='dashed', label='Deep ESN',marker='X',markersize=10,capsize=4)
    # axia.plot(rte,color='red', linewidth=1.0, linestyle='-', label='Deep ESN',marker='^',markersize=10)
    axia.errorbar(xran,ete,yerr=stdete,xerr=None,color='blue', linewidth=1, linestyle='dashed', label='GE Deep ESN',marker='P',markersize=10,capsize=4)
    # axia.plot(ete,color='blue', linewidth=1.0, linestyle='-', label='GE Deep ESN',marker='P',markersize=10)    
    axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    axia.tick_params(labelsize= fs) # 设置坐标轴上刻度的字体大小
    alabarx=[str(j+1) for j in range(G_num)]
    plt.xticks(range(G_num),labels=alabarx)
    # axia.set_xticklabels(['']+alabarx) 
    plt.savefig(repdir+'/'+str(m)+'_CompareASE.eps')
    fig.show()    
    fig.clear()    
    
    # fig=plt.figure(2,  figsize=(12, 8)) 
    # axia=plt.gca()
    # axia.plot(range(10),rtr,color='red', linewidth=1.0, linestyle='-', label='Deep ESN',marker='^',markersize=10)
    # axia.plot(range(10),etr,color='blue', linewidth=1.0, linestyle='-', label='GE Deep ESN',marker='*',markersize=10)    
    # axia.legend(loc=0, fontsize =  fs) # 显示图例，loc=0表示图例会根据图片情况自动摆放
    # plt.savefig(repdir+'/_Compare_NRMSE_train.eps')
    # fig.show()       
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
    axia.plot(range(10),rtr,color='red', linewidth=1.0, linestyle='-', label='Unpuned Deep ESN',marker='^',markersize=10)
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
    axia.plot(range(10),rte,color='red', linewidth=1.0, linestyle='-', label='Unpuned Deep ESN',marker='^',markersize=10)
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
    axia.plot(range(10),ra,color='red', linewidth=1.0, linestyle='-', label='Unpuned Deep ESN',marker='^',markersize=10)
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

if __name__ == '__main__':
   
    
    for m in range(3,4):
        Lredir=Predir+'m'+str(m)+'/'
        # GUILayerMerge_aveg(Lredir)
        AverlizeOVerall(Lredir)
        LayerMerge_aveg(Lredir,equal=1)
    end=0
    
    repdir=Predir+'rep'+'/'
    for m in range(3,4):
        evadir=Predir+'m'+str(m)+'/'
        ComPareRepAndEva(repdir,evadir,m)
        ComPareRepAndEvaASE(repdir,evadir,m)
    end=1
    
    for m in range(3,4):
        evadir=Predir+'m'+str(m)+'/'

        fname='MAX_corr.txt'
        # fname='AMI.txt'
        print('m='+str(m))
        err=anagrowlayer(evadir,fname,pusai)
        print(str(err)+'\n')
    end=2    
    Lredir=Predir+'m3/'

    GUILayerMerge_aveg(Lredir)