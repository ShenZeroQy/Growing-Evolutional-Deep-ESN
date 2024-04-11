#for cuda
#always use float32 to ca
from numpy import * 

# def SMEstimater(X_train:ndarray,method=0):
def SMEstimater(X_train,X_num=50,T_num=3000,method=0):
  #X_train ndarrray[N_num*T_num]
  N_num,Tnum=X_train.shape
  result=zeros((N_num,N_num))
  if (method==0):
    for i in range(N_num):    
      for j in range(i+1,N_num):
        nodei=X_train[i,:]
        nodej=X_train[j,:]
        dis=(nodei-nodej)**2
        dif=1/(1+sum(dis))
        result[i,j]=dif
      
  return result

def EntropyEstimater(X_train,X_num=50,T_num=3000):
  return 0.25