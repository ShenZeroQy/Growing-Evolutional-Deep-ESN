#for cuda
#always use float32 to calculate
from pycuda import driver
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.cumath as gpumath
#end cuda

import numpy as np 



#defination of GPU function
GPUMod = SourceModule("""
//cuda
__global__ void GPU_Hisg_1dim_X(int N_num,int T_num,float*Xtrain,float* result,int mesh,float max){
    //Xtrain:N_num*T_num(rank)
    //result:N_num*mesh
    //call block=N_num,mesh,1
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if(idx>=N_num||idy>=mesh)
      return;
    float rnlow=(float)idy/mesh;
    rnlow=max*(rnlow-0.5);
    // rnlow=0;
    float rnhigh=(float)(idy+1.0)/mesh;
    rnhigh=max*(rnhigh-0.5);
    // rnhigh=1;
    int i;
    int cnt;
    cnt=0;
    for(i=0;i<T_num;i++)
    {
      if( Xtrain[T_num*idx+i]>=rnlow &&  Xtrain[T_num*idx+i]<rnhigh)
      {
        cnt++;
      }
    }
    result[idx*mesh+idy]=(float)cnt/T_num;
    // result[idx*mesh+idy]=rnhigh;

    return;
}
__global__ void GPU_Hisg_1dim_Y(int Y_dim,int T_num,float*Ytrain,float* result,int mesh,float max){
   //Ytrain:Y_dim*T_num(rank)
    //result:Y_dim*mesh
    //call block=Y_dim,mesh,1
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if(idx>=Y_dim||idy>=mesh)
      return;
    float rnlow=(float)idy/mesh;
    rnlow=max*(rnlow-0.5);
    // rnlow=0;
    float rnhigh=(float)(idy+1.0)/mesh;
    rnhigh=max*(rnhigh-0.5);
    // rnhigh=1;
    int i;
    int cnt;
    cnt=0;
    for(i=0;i<T_num;i++)
    {
      if( Ytrain[T_num*idx+i]>=rnlow &&  Ytrain[T_num*idx+i]<rnhigh)
      {
        cnt++;
      }

    }
    result[idx*mesh+idy]=(float)cnt/T_num;
    return;
}
__global__ void GPU_Hisg_2dim_XY(int N_num,int Y_dim,int T_num,float*Xtrain,float*Ytrain,float* result,int meshx,float maxx,float maxy){
    //Xtrain:N_num*T_num(rank)
    //Only deal one neuron :N_index 
    //Ytrain:Y_dim*T_num(rank)
    //result:(meshx*meshy[meshID])*N_num
    //meshID=idy*meshx+idx
    //call block=meshx,meshy,N_num
    //meshX==meshY
    int idm = threadIdx.x + blockDim.x * blockIdx.x;
    int idx=idm/meshx;
    int idy=idm%meshx;
    int idz = threadIdx.y + blockDim.y * blockIdx.y;
    int meshy=meshx;
    if(idx>=meshx||idy>=meshy||idz>=N_num)
      return;


    float rnlowx=(float)idx/meshx;
    rnlowx=maxx*(rnlowx-0.5);
    float rnhighx=(float)(idx+1.0)/meshx;
    rnhighx=maxx*(rnhighx-0.5);

    float rnlowy=(float)idy/meshy;
    rnlowy=maxy*(rnlowy-0.5);
    float rnhighy=(float)(idy+1.0)/meshy;
    rnhighy=maxy*(rnhighy-0.5);
    
    int jx,jy,jt;
    int cnt=0;
    jx=idz;
    for (jt=0;jt<T_num;jt++)
      if( Xtrain[jx*T_num+jt]>=rnlowx &&  Xtrain[jx*T_num+jt]<rnhighx)
      {
        for(jy=0;jy<Y_dim;jy++)
          if( Ytrain[jy*T_num+jt]>=rnlowy &&  Ytrain[jy*T_num+jt]<rnhighy)
          {
          cnt++;
          }
      }
    int meshID=idy*meshx+idx;
    result[meshID*N_num+idz]=(float)cnt/(T_num*Y_dim);
    return;
}

//!cuda
  """)



def IXY(X_train,Y_train,X_num=50,Y_dim=1,T_num=3000):
  #X_train ndarrray[N_num*T_num]
  #Y_train ndarrray[Y_dim*T_num]

  X_train=X_train.astype(np.float32)
  Y_train=Y_train.astype(np.float32)

  meshX=128
  meshY=128 
  resultX=np.zeros((X_num,meshX))
  resultX=resultX.astype(np.float32)
  
  resultY=np.zeros((Y_dim,meshX))
  resultY=resultY.astype(np.float32)   

 
  maxX=np.amax(abs(X_train))*1.01
  maxY=np.amax(abs(Y_train))*1.01

  #Compute H(X)
  X_train_gpu=gpuarray.to_gpu(X_train)
  GPUfunc=GPUMod.get_function("GPU_Hisg_1dim_X")
  #__global__ void GPU_Hisg_1dim_X(int N_num,int T_num,float*Xtrain,float* result,int mesh,float max)
  GPUfunc(np.int32(X_num),np.int32(T_num),X_train_gpu,driver.Out(resultX),np.int32(meshX),np.float32(maxX),block=(32,32,1),grid=(int(X_num/32)+1,int(meshX/32)+1,1))
  # PX=np.mean(resultX,axis=0)
  # HX=0
  # for px in PX:
  #   if px>0:
  #       HX=HX-px*np.log(px)
  # Hes=HX
  
  #Compute H(Y)
  Y_train_gpu=gpuarray.to_gpu(Y_train)
  GPUfunc=GPUMod.get_function("GPU_Hisg_1dim_Y")
  # #___global__ void GPU_Hisg_1dim_Y(int Y_dim,int T_num,float*Ytrain,float* result,int mesh,float max)
  GPUfunc(np.int32(Y_dim),np.int32(T_num),Y_train_gpu,driver.Out(resultY),np.int32(meshY),np.float32(maxY),block=(1,32,1),grid=(Y_dim,int(meshX/32)+1,1))
  PY=np.mean(resultY,axis=0)
  PY=PY/sum(PY)
  HY=0
  for py in PY:
    if py>0:
        HY=HY-py*np.log(py)
  # Hes=HY

  #Compute I(XY)
  meshX=32 
  resultXY=np.zeros((meshX*meshX,X_num))
  resultXY=resultXY.astype(np.float32) 
   
  GPUfunc=GPUMod.get_function("GPU_Hisg_2dim_XY")
  # _global__ void GPU_Hisg_2dim_XY(int N_num,int Y_dim,int T_num,float*Ytrain,float* result,int meshx,float maxx,int mashy,float maxy)
  GPUfunc(np.int32(X_num),np.int32(Y_dim),np.int32(T_num),X_train_gpu,Y_train_gpu,driver.Out(resultXY),
  np.int32(meshX),np.float32(maxX),np.float32(maxY),block=(32,32,1),grid=(int(meshX*meshX/32)+1,int(X_num/32)+1,1))
  resultXY=resultXY.T

  MI=np.zeros((X_num,),dtype=np.float32)
  for i in range(X_num):
    PX=resultX[i,:]
    PX=PX/sum(PX)
    HX=0
    for px in PX:
      if px>0:
        HX=HX-px*np.log(px)
    PXY=resultXY[i,:]
    PXY=PXY/sum(PXY)
    IXY=0
    for pxy in PXY:
      if pxy>0:
        IXY=IXY-pxy*np.log(pxy)
   
    Hes=(HX+HY-IXY)/np.sqrt(HX*HY)
    if(HX*HY==0):
      Hes=0 
    MI[i]=Hes


  AMI=np.mean(MI)
  # AMI=AMI*X_num
  # Hes=IXY
  # result=[HX,HY,IXY] 

  # return result.astype(np.float64)
  return AMI.astype(np.float64)

