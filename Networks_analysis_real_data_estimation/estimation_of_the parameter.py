
import numpy as np
import os 
import pandas as pd
from src.utils import *
from src.stiefel import *
from src import mcmc, saem
import pickle

seed=22
np.random.seed(seed)
path_save_param = "param"
try:
    os.mkdir(path_save_param)
except:
    pass
path_save = "result"
try:
    os.mkdir(path_save)
except:
    pass

d=25

#path to the folder with networks data
path_load_data="tofill"
#Networks real dataset on fmri
netmat = np.loadtxt(os.path.join(path_load_data,"netmats2_d"+str(d)+".txt"), comments="#", delimiter=" ", unpack=False)

netmat=np.reshape(netmat,(812,d,-1))
n=d
p=5
n_iter=1000


for slice in [True, False]:
    # Initialize the parameters at randomly chosen values
    print("Initializing the MCMC-SAEM algorithm.")
    F = 5*np.random.randn(n,p)
    mu = np.random.randn(p)

    prop_l = 0.1
    if slice:
        prop_X=1
        m=10
        # Perform initial MCMC steps on X and lambda
        histo,Xs_mh, ls_mh, _, _, _ = mcmc.slice_rieman(netmat, (F, mu, 1, 1), n_iter=200, init=None, prop_X=prop_X, prop_l=prop_l,m=m)
    else:# n_iter x 4
        prop_X=0.01
        histo,Xs_mh, ls_mh, _, _, _ = mcmc.mh(netmat, (F, mu, 1, 1), n_iter=800, init=None, prop_X=prop_X, prop_l=prop_l)

    print("Running the MCMC-SAEM algorithm.")
    if slice:
        result = saem.mcmc_saem(netmat, Xs_mh, ls_mh, (F, mu, 1, 1), n_iter=n_iter, n_mcmc=20,prop_X=prop_X,slice_rieman=slice,use_adapt=False,m=m)
    else:# n_mcmc x 4
        result = saem.mcmc_saem(netmat, Xs_mh, ls_mh, (F, mu, 1, 1), n_iter=n_iter, n_mcmc=80,prop_X=prop_X,slice_rieman=slice,use_adapt=True,m=m)
    theta = result["theta"]
    Xs_mh = result["Xs_mh"]
    ls_mh = result["ls_mh"]
    lks=result["history"]["lks"]
    df_lks=pd.DataFrame(np.array(lks))
    
    #SAVE
    if slice:
        df_lks.to_csv(os.path.join(path_save,"slice_estimation_real_lks_n_p{:.0f},{:.0f}".format(n,p)))
        with open(os.path.join(path_save_param,"slice_estimation_real_param_n_p{:.0f},{:.0f}".format(n,p)), 'wb') as f:
            pickle.dump(theta, f)

        with open(os.path.join(path_save_param,"slice_estimation_real_Xs_n_p{:.0f},{:.0f}".format(n,p)), 'wb') as f:
            pickle.dump(Xs_mh, f)

        with open(os.path.join(path_save_param,"slice_estimation_real_ls_n_p{:.0f},{:.0f}".format(n,p)), 'wb') as f:
            pickle.dump(ls_mh, f)
    else:
        df_lks.to_csv(os.path.join(path_save,"mhadapt_estimation_real_lks_n_p{:.0f},{:.0f}".format(n,p)))
        with open(os.path.join(path_save_param,"mhadapt_estimation_real_param_n_p{:.0f},{:.0f}".format(n,p)), 'wb') as f:
            pickle.dump(theta, f)
        with open(os.path.join(path_save_param,"mhadapt_estimation_real_Xs_n_p{:.0f},{:.0f}".format(n,p)), 'wb') as f:
            pickle.dump(Xs_mh, f)

        with open(os.path.join(path_save_param,"mhadapt_estimation_real_ls_n_p{:.0f},{:.0f}".format(n,p)), 'wb') as f:
            pickle.dump(ls_mh, f)

