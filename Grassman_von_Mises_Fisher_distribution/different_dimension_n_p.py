import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl
from Sampler_grassman import *
from evaluation import *
import time
import pandas as pd
import os 
import itertools
from Hybrid_monte_carlo import *
path_save="Code_to_git/Grassman_von_Mises_Fisher_distribution/results"

seed=22
np.random.seed(seed)

N=100000

n_resamp=10

np_list=[(3,2),(30,20),(100,2)]
w_list=[7]
to_prod=[np_list,w_list]
list_param=list(itertools.product(*to_prod))

#Only useful for projection
log_prob=lambda x:1
n,p=3,2
to_reproject=Sampler_Slice_Sampling_Grassman(log_prob,n,p)


for param in list_param:
    (n,p),w=param

    #Parameter and vMF
    U= np.zeros((n,n))
    U[:p,:p]=np.eye(p)
    log_prob = lambda x : np.einsum('ik,ik->k',U,x@x.T).sum()

    #initialisation
    X_0 = np.random.randn(n,p)
    X_0= to_reproject.proj_V(X_0)
    
    eval=evaluator(200)
    
    print("slice")

    A_slice=np.zeros(n_resamp)
    A_slice_rate=np.zeros(n_resamp)
    for i in range(n_resamp):
        sampler_grass=Sampler_Slice_Sampling_Grassman(log_prob,n,p)
        t1=time.time()
        data,rate = sampler_grass.run_kernel(X_0, N,w=w,use_adapt=False,m=1)
        t2=time.time()
        delta=t2-t1
        ess=eval.ESS(np.array([np.einsum('ik,ik->k',U,x@x.T).sum()  for x in data]))
        A_slice[i]=ess
        A_slice_rate[i]=rate

    print("mean ess", np.mean(A_slice))
    print("std ess", np.std(A_slice))
    dh=pd.DataFrame(A_slice)
    rate=pd.DataFrame(A_slice_rate)
    
    #save
    name="n_toys_examples_slice_grassman_s{:.0f}_w{:.0f}_n{:.0f}_p{:.0f}_N{:.0f}.csv".format(seed,w,n,p,N)
    dh.to_csv(os.path.join(path_save,name))
    name="n_rate_toys_examples_slice_grassman_s{:.0f}_w{:.0f}_n{:.0f}_p{:.0f}_N{:.0f}.csv".format(seed,w,n,p,N)
    rate.to_csv(os.path.join(path_save,name))

print("HMC")

h_list=[0.01,1,2]

to_prod=[np_list,h_list]
list_param=list(itertools.product(*to_prod))

T=1
for param in list_param:
    (n,p),h=param
    
    #Parameter and vMF
    U= np.zeros((n,n))
    U[:p,:p]=np.eye(p)
    log_prob = lambda x : np.einsum('ik,ik->k',U,x@x.T).sum()
    grad_log_prob = lambda x: 2*U@x

    #initialisation
    X_0 = np.random.randn(n,p)
    X_0= to_reproject.proj_V(X_0)
    
    eval=evaluator(200)  
    A_mh=np.zeros(n_resamp)
    for i in range(n_resamp):
        sampler_hmc=Hybrid_Monte_Carlo_Grassman(log_prob=log_prob,grad_log_prob=grad_log_prob,n=n,p=p,T=T,h=h)
        t1=time.time()
        samples = sampler_hmc.run_kernel(X_0, N)
        t2=time.time()
        delta=t2-t1

        ess=eval.ESS(np.array([np.einsum('ik,ik->k',U,x@x.T).sum() for x in samples]))
        A_mh[i]=ess

    print("mean ess", np.mean(A_mh))
    print("std ess", np.std(A_mh))
    dg=pd.DataFrame(A_mh)
    
    #save
    name="n_toys_examples_grassman_hmc_s{:.0f}_h{:.2f}_n{:.0f}_p{:.0f}_N{:.0f}.csv".format(seed,h,n,p,N)
    dg.to_csv(os.path.join(path_save,name))

