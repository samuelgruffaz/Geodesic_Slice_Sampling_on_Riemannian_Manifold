"""
-
This script reproduces the experiment of section 3.2 on parameter estimation in dimension (30,5).
The constants are named as follow:
- n_samples is the number of model samples. In this experiment, n_samples=100.
- n is the number of nodes. In this experiment, n=30.
- p is the number of orthonormal columns. In this experiment, p=5. (in the paper we denote by k)

The parameter theta is composed of:
- F and mu designate the vMF parameter and the mean eigenvalues
- sigma and sigma_l in the code correspond to sigma_epsilon and sigma_lambda in the paper

The variables used in the code are:
- Xs (n_samples,n,p) is the list of vMF samples (X_1, ..., X_N)
- ls (n_samples,p) is the list of patterns amplitudes for each individual (lambda_1, ..., lambda_N)
- Xs_mh and ls_mh are the MCMC estimate of Xs and ls once the MCMC has converged
"""

import numpy as np
from matplotlib.pyplot import *
import os
import pandas as pd
from src.utils import *
from src.stiefel import *
from src import spa, vmf, mcmc, model, saem
import os
import pickle
import itertools

np.random.seed(0)

path_save = "results_est"
try:
    os.mkdir(path_save)
except:
    pass



adapt=False
#==============================
# EXPERIMENT 1 : SMALL NOISE
#==============================

n,p=30,5

print("Experiment: parameter estimation with small noise.")
print("Generating the synthetic data set.")
n_repet=10
n_iter=100

ani_list=[1,100]
slice_bool=[True,False]
to_prod=[ani_list,slice_bool]
list_param=list(itertools.product(*to_prod))
for param in list_param:
    ani,slice=param
    A_norm_F=np.zeros((n_repet,p))# a changer
    A_norm_mu=np.zeros(n_repet)

    A_lks=np.zeros((n_repet,n_iter))
    A_conv_true=np.zeros((n_repet,n_iter,2))
    
    for i in range(n_repet):
        
        #true parameter 
        F0 = ([ani]+(p-1)*[1])*unif(n,p)# uniform sampling on the Stiefel nxp and rescaling with ani
        mu0 = np.array([10.]+(p-1)*[2.])
        mu0 = mu0[np.abs(mu0).argsort()][::-1]
        sigma0 = 0.1 # sigma_epsilon
        sigma_l0 = 2 # sigma_lambda

        theta0 = (F0, mu0, sigma0, sigma_l0)

        #generation of the synthetic samples
        n_samples = 100
        ls = mu0[None,:].repeat(n_samples, axis=0)
        ls += sigma_l0*np.random.randn(*ls.shape)
        Xs = vmf.sample_von_mises_fisher_slice(F0, n_iter=n_samples, burn=10000, stride=100)
        
        As = comp(Xs, ls)
        idx = np.triu_indices(F0.shape[0])
        noise = sigma0*np.random.randn(*As.shape)
        noise = (noise+noise.transpose(0,2,1))/np.sqrt(2)
        As += noise

        #estimation from the samples
        # Initialize the parameters at randomly chosen values
        print("Initializing the MCMC-SAEM algorithm.")
        F = 5*np.random.randn(n,p)
        mu = np.random.randn(p)
        

        # Perform initial MCMC steps on X and lambda
    
        prop_l = 0.1
        if slice:
            prop_X=1# prop_X is the parameter w
            histo,Xs_mh, ls_mh, _, _, _ = mcmc.slice_rieman(As, (F, mu, 1, 1), n_iter=200, init=None, prop_X=prop_X, prop_l=prop_l)
        else:# n_iter x 4
            prop_X=0.01
            histo,Xs_mh, ls_mh, _, _, _ = mcmc.mh(As, (F, mu, 1, 1), n_iter=800, init=None, prop_X=prop_X, prop_l=prop_l)
            

        # Run the MCMC-SAEM for 100 iterations with 20 MCMC steps per SAEM step
        print("Running the MCMC-SAEM algorithm.")
        if slice:
            result = saem.mcmc_saem(As, Xs_mh, ls_mh, (F, mu, 1, 1), n_iter=n_iter, n_mcmc=20,prop_X=prop_X,slice_rieman=slice,use_adapt=adapt,m=5)
        else:# n_mcmc x 4
            result = saem.mcmc_saem(As, Xs_mh, ls_mh, (F, mu, 1, 1), n_iter=n_iter, n_mcmc=80,prop_X=prop_X,slice_rieman=slice,use_adapt=adapt)


    #==============================
    # DISPLAY THE RESULTS
    #==============================

        # Open results with small noise
        theta = result["theta"]
        Xs_mh = result["Xs_mh"]
        ls_mh = result["ls_mh"]
        lks=result["history"]["lks"]
        Flist=result["history"]["F"]
        mulist=result["history"]["mu"]
        
        for itera in range(1,len(Flist)):
            #we permute the indices to match the true parameter
            Xs_mh, _, (F, mu, _, _), m, s = saem.map_to_ground_truth(Xs_mh, ls_mh, (Flist[itera],mulist[itera],1,1), theta0)
            #we compare the distance to the rRMSE to the true parameter
            A_conv_true[i,itera-1,0]=np.linalg.norm(F.copy()-F0)/np.linalg.norm(F0)
            A_conv_true[i,itera-1,1]=np.linalg.norm(mu.copy()-mu0)/np.linalg.norm(mu0)

        
        # Align the signs of the columns of F to the ground truth to ease visualization
        Xs_mh, _, (F, mu, _, _), m, s = saem.map_to_ground_truth(Xs_mh, ls_mh, theta, theta0)
        A_lks[i]=np.array(lks)
        
        A_norm_F[i]=np.linalg.norm(F, axis=0)
        A_norm_mu[i]=np.linalg.norm(mu-mu0)/np.linalg.norm(mu0)

    df_lks=pd.DataFrame(A_lks)
    
    df_norm=pd.DataFrame(A_norm_F)
    df_mu=pd.DataFrame(A_norm_mu)
    #SAVE
    if slice:
        df_lks.to_csv(os.path.join(path_save,"slice_estimation_lks_ani{:.0f}_n_p{:.0f},{:.0f}".format(ani,n,p)))
        with open(os.path.join(path_save,"slice_estimation_conv_true_ani{:.0f}_n_p{:.0f},{:.0f}".format(ani,n,p)), 'wb') as f:
            pickle.dump(A_conv_true, f)

        df_norm.to_csv(os.path.join(path_save,"slice_estimation_rel_fnorm_ani{:.0f}_n_p{:.0f},{:.0f}".format(ani,n,p)))
        df_mu.to_csv(os.path.join(path_save,"slice_estimation_rel_munorm_ani{:.0f}_n_p{:.0f},{:.0f}".format(ani,n,p)))
    else:
        df_lks.to_csv(os.path.join(path_save,"mhadapt_estimation_lks_ani{:.0f}_n_p{:.0f},{:.0f}".format(ani,n,p)))
        with open(os.path.join(path_save,"mhadapt_estimation_conv_true_ani{:.0f}_n_p{:.0f},{:.0f}".format(ani,n,p)), 'wb') as f:
            pickle.dump(A_conv_true, f)
        df_norm.to_csv(os.path.join(path_save,"mhadapt_estimation_rel_fnorm_ani{:.0f}_n_p{:.0f},{:.0f}".format(ani,n,p)))
        df_mu.to_csv(os.path.join(path_save,"mhadapt_estimation_rel_munorm_ani{:.0f}_n_p{:.0f},{:.0f}".format(ani,n,p)))

