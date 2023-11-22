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
np.random.seed(seed)

N=100000

n_resamp=10

w_list=[1,3,5,7,9,11]
np_list=[(3,2),(30,2),(100,2),(30,5),(30,10),(30,20)]
to_prod=[np_list,w_list]
list_param=list(itertools.product(*to_prod))

#to use projection operator
log_prob=lambda x:1
p=2
n=3
to_reproject=Sampler_Slice_Sampling_Geodesics(log_prob,n,p)

print(list_param)

for param in list_param:
    (n,p),w=param

    #vMF Parameter
    F=np.zeros((n,p))
    F[:p,:]=np.diag((np.arange(p)+1))

    log_prob = lambda x : np.einsum('ik,ik->k',F,x).sum()

    #random init (no impact of burn in period)
    X_0 = np.random.randn(n,p)
    X_0= to_reproject.proj_V(X_0)
    
    eval=evaluator(200)
    
    print("slice")

    A_slice=np.zeros(n_resamp)
    A_slice_rate=np.zeros(n_resamp)
    for i in range(n_resamp):
        sampler_slice=Sampler_Slice_Sampling_Geodesics(log_prob,n,p)
        t1=time.time()
        data = sampler_slice.run_kernel(X_0, N,w=w,use_adapt=False,m=1)
        t2=time.time()
        delta=t2-t1
        ess=eval.ESS(np.array([np.einsum('ik,ik->k',F,x).sum()  for x in data]))
        A_slice[i]=ess
        A_slice_rate[i]=sampler_slice.rate

    print("mean ess", np.mean(A_slice))
    print("std ess", np.std(A_slice))
    dh=pd.DataFrame(A_slice)
    
    
    name="toys_examples_slice_s{:.0f}_w{:.0f}_n{:.0f}_p{:.0f}_N{:.0f}.csv".format(seed,w,n,p,N)
    dh.to_csv(os.path.join(path_save,name))

print("mh")
for npp in np_list: 
    (n,p)=npp

    #vMF Parameter
    F=np.zeros((n,p))
    F[:p,:]=np.diag((np.arange(p)+1))

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

        
    name="toys_examples_mh_s{:.0f}_n{:.0f}_p{:.0f}_N{:.0f}.csv".format(seed,n,p,N)
    dg.to_csv(os.path.join(path_save,name))

print("geo_mh")
for npp in np_list: 
    (n,p)=npp
   
    #vMF Parameter
    F=np.zeros((n,p))
    F[:p,:]=np.diag((np.arange(p)+1))

    log_prob = lambda x : np.einsum('ik,ik->k',F,x).sum()

    #random init (no impact of burn in period)
    X_0 = np.random.randn(n,p)
    X_0= to_reproject.proj_V(X_0)
    
    eval=evaluator(200)  
    A_mh=np.zeros(n_resamp)
    for i in range(n_resamp):
        sampler_mh=Sampler_geo_mh(log_prob,n,p)
        t1=time.time()
        samples = sampler_mh.run_kernel(X_0, N)
        t2=time.time()
        delta=t2-t1
    
        ess=eval.ESS(np.array([np.einsum('ik,ik->k',F,x).sum() for x in samples]))
        A_mh[i]=ess

    print("mean ess", np.mean(A_mh))
    print("std ess", np.std(A_mh))
    dg=pd.DataFrame(A_mh)
    
        
    name="toys_examples_geo_mh_s{:.0f}_n{:.0f}_p{:.0f}_N{:.0f}.csv".format(seed,n,p,N)
    dg.to_csv(os.path.join(path_save,name))

