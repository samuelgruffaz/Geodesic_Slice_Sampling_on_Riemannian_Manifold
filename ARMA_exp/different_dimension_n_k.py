import numpy as np
import matplotlib.pyplot as plt

from evaluation import *
import time
import pandas as pd
import os 
from ARMA import *
from Sampler_class import *
path_save="results"

seed=22
np.random.seed(seed)

N=100000

n_resamp=10


#for projection on the stiefel manifold
log_prob=lambda x:1
n,p=3,2
to_reproject=Sampler_Slice_Sampling_Geodesics(log_prob,n,p)


#dimension of the problem
p=2
n=100
T=10 # number of observations

B=np.random.randn(p,p)# transition matrix on latent variabled
B=B/np.sqrt(np.sum(B**2,axis=1))[:,None]# line sum to 1
sigma=0.5
Q=sigma*np.eye(p)# latent noise covariance
R=2*np.eye(n)+np.ones((n,n))/n# observation noise covariance
R=R/10 #reduction of the noise
F=np.random.randn(n,p)
F=to_reproject.proj_V(F)#random stiefel prior
arma=ARMA(R,Q,B,F,p,n)
H_gen=F+0.1*np.random.randn(n,p)
H_gen=to_reproject.proj_V(H_gen)# Observation matrix for generation (not to far from the prior)
X,Z=arma.generator(T,H_gen)# we generate the observation Z and the latent variable X
log_prob=arma.log_prob(Z)#renvoir p(H|Z)

X_0 = np.random.randn(n,p)
X_0= to_reproject.proj_V(X_0)#initialization
list_param=[(1,10),(1,1),(2,5)]

for param in list_param:
    m,w=param

    eval=evaluator(200)
    
    print("slice")

    A_slice=np.zeros(n_resamp)
    A_slice_rate=np.zeros(n_resamp)
    for i in range(n_resamp):
        sampler=Sampler_Slice_Sampling_Geodesics(log_prob,n,p)
        t1=time.time()
        data,rate = sampler.run_kernel(X_0, N,w=w,use_adapt=False,m=m)
        t2=time.time()
        delta=t2-t1
        ess=eval.ESS(np.array([np.einsum('ik,ik->k',F,x).sum()  for x in data]))
        A_slice[i]=ess
        print("ESS GSS :",ess)
        A_slice_rate[i]=rate
        print("ESS rate :",rate)
        

    print("mean ess", np.mean(A_slice))
    print("std ess", np.std(A_slice))
    dh=pd.DataFrame(A_slice)
    rate=pd.DataFrame(A_slice_rate)
    
    #save
    name="arma_slice_s{:.0f}_w{:.0f}_n{:.0f}_p{:.0f}_N{:.0f}_m{:.0f}.csv".format(seed,w,n,p,N,m)
    dh.to_csv(os.path.join(path_save,name))
    name="arma_rate_slice_s{:.0f}_w{:.0f}_n{:.0f}_p{:.0f}_N{:.0f}_m{:.0f}.csv".format(seed,w,n,p,N,m)
    rate.to_csv(os.path.join(path_save,name))

print("AMH")


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

name="arma_mh_s{:.0f}_n{:.0f}_p{:.0f}_N{:.0f}.csv".format(seed,n,p,N)
dg.to_csv(os.path.join(path_save,name))

