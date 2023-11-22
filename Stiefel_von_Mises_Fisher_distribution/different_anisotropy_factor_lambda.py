import numpy as np
import matplotlib.pyplot as plt
from Sampler_class import *
from evaluation import *
import time

import pandas as pd
import os 
import itertools

path_save="Stiefel_von_Mises_Fisher_distribution/results"

seed=22
numpy.random.seed(seed)

p=2
n=30
N=100000


n_resamp=10
w_list=[5]
ani_list=[1,10,100]
to_prod=[ani_list,w_list]
list_param=list(itertools.product(*to_prod))

#to use projection operator
log_prob=lambda x:1
to_reproject=Sampler_Slice_Sampling_Geodesics(log_prob,n,p)

k=0

for param in list_param:
    ani,w=param
    
    #vMF Parameter
    F=np.zeros((n,p))
    F[0,0]=1
    F[1,1]=ani

    log_prob = lambda x : np.einsum('ik,ik->k',F,x).sum()

    #random init (no impact of burn in period)
    X_0 = np.random.randn(n,p)
    X_0= to_reproject.proj_V(X_0)
    
    eval=evaluator(200)
    
    print("slice")

    A_slice=np.zeros(n_resamp)
    for i in range(n_resamp):
        sampler_slice=Sampler_Slice_Sampling_Geodesics(log_prob,n,p)
        t1=time.time()
        data = sampler_slice.run_kernel(X_0, N,w=w,use_adapt=False,m=1)
        t2=time.time()
        delta=t2-t1
        ess=eval.ESS(np.array([np.einsum('ik,ik->k',F,x).sum()  for x in data]))
        A_slice[i]=ess

    print("mean ess", np.mean(A_slice))
    print("std ess", np.std(A_slice))
    dh=pd.DataFrame(A_slice)
    
    name="toys_examples_slice_s{:.0f}_w{:.0f}_ani{:.0f}_N{:.0f}_n_p{:.0f},{:.0f}.csv".format(seed,w,ani,N,n,p)
    dh.to_csv(os.path.join(path_save,name))


print("mh")
for ani in ani_list: 
  
    #vMF Parameter
    F=np.zeros((n,p))
    F[0,0]=1
    F[1,1]=ani

    log_prob = lambda x : np.einsum('ik,ik->k',F,x).sum()

    #random init (no impact of burn in period)
    X_0 = np.random.randn(n,p)
    X_0= to_reproject.proj_V(X_0)
    
    eval=evaluator(200)  
    A_mh=np.zeros(n_resamp)
    for i in range(n_resamp):
        sampler_mh=Sampler_mh(log_prob,n,p)
        t1=time.time()
        samples = sampler_mh.run_kernel(X_0, N)
        t2=time.time()
        delta=t2-t1
    
        ess=eval.ESS(np.array([np.einsum('ik,ik->k',F,x).sum() for x in samples]))
        A_mh[i]=ess

    print("mean ess", np.mean(A_mh))
    print("std ess", np.std(A_mh))
    dg=pd.DataFrame(A_mh)
    
        
    name="toys_examples_mh_s{:.0f}_ani{:.0f}_N{:.0f}_n_p{:.0f},{:.0f}.csv".format(seed,ani,N,n,p)
    dg.to_csv(os.path.join(path_save,name))
        

print("geo_mh")
for ani in ani_list: 
   
    #vMF Parameter
    F=np.zeros((n,p))
    F[0,0]=1
    F[1,1]=ani

    log_prob = lambda x : np.einsum('ik,ik->k',F,x).sum()

    #random init (no impact of burn in period)
    X_0 = np.random.randn(n,p)
    X_0= to_reproject.proj_V(X_0)
    eval=evaluator(200)  
    A_mh=np.zeros(n_resamp)
    for i in range(n_resamp):
        sampler_mh=Sampler_mh(log_prob,n,p)
        t1=time.time()
        samples = sampler_mh.run_kernel(X_0, N)
        t2=time.time()
        delta=t2-t1
    
        ess=eval.ESS(np.array([np.einsum('ik,ik->k',F,x).sum() for x in samples]))
        A_mh[i]=ess

    print("mean ess", np.mean(A_mh))
    print("std ess", np.std(A_mh))
    dg=pd.DataFrame(A_mh)
    
    name="toys_examples_geo_mh_s{:.0f}_ani{:.0f}_N{:.0f}_n_p{:.0f},{:.0f}.csv".format(seed,ani,N,n,p)
    dg.to_csv(os.path.join(path_save,name))

