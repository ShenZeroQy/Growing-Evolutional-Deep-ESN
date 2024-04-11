from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import loadtxt


trainl=2000
testl=3500

data=loadtxt('./data/MackeyGlass.txt')
rcParams['font.size'] = 12 
figure(1,figsize=(11,4))
axis=gca()
axis.set_xlim(0,testl+500)
axis.plot( range(0,200),data[0:200], color='g',linewidth=0.2,label='Washout')
axis.plot( range(200,2001),data[200:2001], color='b',linewidth=0.2,label='Train')
axis.plot( range(2000,testl),data[2000:testl], color='r',linewidth=0.2,label='Test')
axis.legend(loc=4)
xlabel('t')
savefig('./data/MG-31-11.eps')    
figure(1).show()
close(figure(1))

data=loadtxt('./data/LaterSunblack.txt')

rcParams['font.size'] = 10 
figure(2,figsize=(11,4))
axis=gca()
axis.set_xlim(0,4700)
axis.set_ylim(0,300)
axis.plot( range(0,200),data[0:200], color='g',linewidth=0.2,label='Washout')
axis.plot( range(200,2000),data[200:2000], color='b',linewidth=0.2,label='Train')
axis.plot( range(2000,4017),data[2000:4017], color='r',linewidth=0.2,label='Test')
axis.legend(loc=4)
xlabel('t/day')
axins2=inset_axes(axis,width="50%", height="40%", loc=1)
axins2.plot( range(1950,2001),data[1950:2001], color='b',linewidth=0.2)
axins2.plot( range(2000,2050),data[2000:2050], color='r',linewidth=0.2)
savefig('./data/SS-d.eps')    

figure(2).show()
close(figure(2))

data=loadtxt('./data/MackeySine.txt')

rcParams['font.size'] = 10 
figure(3,figsize=(11,4))
axis=gca()
axis.set_xlim(0,testl+500)
# axis.set_ylim(-0.6,0.6)
axis.plot( range(0,200),data[0:200], color='g',linewidth=0.2,label='Washout')
axis.plot( range(200,trainl+1),data[200:trainl+1], color='b',linewidth=0.2,label='Train')
axis.plot( range(trainl,testl),data[trainl:testl], color='r',linewidth=0.2,label='Test')
axis.legend(loc=4)
xlabel('t')
# axins2=inset_axes(axis,width="50%", height="40%", loc=1)
# axins2.plot( range(1950,2001),data[1950:2001], color='b',linewidth=0.2)
# axins2.plot( range(2000,2050),data[2000:2050], color='r',linewidth=0.2)
savefig('./data/SS-m.eps')    

figure(3).show()
close(figure(3))

