"""
This file contains the following functions.

Three functions (designed for both models with weighted and binary coefficients) implement the MCMC steps of the MCMC-SAEM.
- mh        : performs MCMC the steps of the MCMC-SAEM algorithm based on Metropolis-Hastings transitions
- mala      : fulfills the same task, but uses the Metropolis-Adjusted Langevin Dynamics
                algorithm instead of Metropolis-Hastings. The mala function is overall slower,
                but in some high-dimensional cases using it instead of mh accelerates the
                global convergence.
- mh_cluster: perform MCMC steps of the MCMC-SAEM for the mixture model

Two functions perform missing link imputation.
- mh_mask   : run a MCMC to sample from the posterior distribution of unknown coefficients given the known coefficients.
- map_mask  : computes the MAP estimator of the unknown coefficients given the known coefficients.

The parameter theta is composed of:
- F and mu designate the vMF parameter and the mean eigenvalues
- sigma and sigma_l in the code correspond to sigma_epsilon and sigma_lambda in the paper

The variables used in the code are:
- As (n_samples,n,n) is the list of adjacency matrices
- Xs (n_samples,n,p) is the list of vMF samples (X_1, ..., X_N)
- ls (n_samples,p) is the list of patterns amplitudes for each individual (lambda_1, ..., lambda_N)
"""


import numpy as np
from tqdm.auto import *
from numba import njit, prange

import src.stiefel as st
from src import spa, model, model_bin, model_cluster, model_bin_cluster, sampler_slice_rieman
#from Code.Sampler import *

norm = np.linalg.norm

@njit
def mh_geo(As, theta, n_iter, init=None, prop_X=0.01, prop_l=0.5, setting="gaussian",m=1):
    """
    Metropolis within Gibbs sampler for the base model.
    - setting can be set to "binary" to handle binary networks
    - prop_X and prop_l are the proposal variances for X and l
    The function returns the final values of X and l, as well as the running likelihood
    and the chain acceptance rates.
    """
    F, mu, sigma, sigma_l = theta
    n_samples = As.shape[0]
    accepts_X = np.zeros((n_iter, n_samples))
    accepts_l = np.zeros((n_iter, n_samples))
    n, p = F.shape[-2:]
    X_history=[]
    if init==None:
        mode = st.proj_V(F)
        Xs = np.zeros((n_samples, n, p))
        ls = sigma_l*np.random.randn(n_samples, p)
        for i in range(n_samples):
            Xs[i] = mode
            ls[i] += mu
    else:
        Xs, ls = init
        Xs = Xs.copy()
        ls = ls.copy()
    
    if setting=="gaussian":
        current_log_lk = np.array([model.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    elif setting=="binary":
        current_log_lk = np.array([model_bin.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    
    for t in range(n_iter):
        X_history.append(Xs[0].copy())
        for i in range(n_samples):
            
            #sample the direction of the geodesics
            
            A,Q,R,v_l,e,v_r = sampler_slice_rieman.sample_tangent_sphere_geo(Xs[i])
            X2 = sampler_slice_rieman.walk_geodesic(Xs[i],A,Q,R,v_l,e,v_r,prop_X)
            if setting=="gaussian":
                new_log_lk = model.log_lk_partial(X2, ls[i], As[i], theta)
            elif setting=="binary":
                new_log_lk = model_bin.log_lk_partial(X2, ls[i], As[i], theta)
            log_alpha = new_log_lk - current_log_lk[i]
            # [X] Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                Xs[i] = X2
                current_log_lk[i] = new_log_lk
                accepts_X[t,i] = 1
            else:
                accepts_X[t,i] = 0
            
            #update the log_lk
            current_log_lk[i] = new_log_lk
            
            
            # [l] Generate next move
            l2 = ls[i] + prop_l*np.random.randn(p)
            # [l] Compute the acceptance log-probability
            if setting=="gaussian":
                new_log_lk = model.log_lk_partial(Xs[i], l2, As[i], theta)
            elif setting=="binary":
                new_log_lk = model_bin.log_lk_partial(Xs[i], l2, As[i], theta)
            log_alpha = new_log_lk - current_log_lk[i]
            # [l] Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                ls[i] = l2
                current_log_lk[i] = new_log_lk
                accepts_l[t,i] = 1
            else:
                accepts_l[t,i] = 0
            
    return X_history,Xs, ls, current_log_lk.sum(), accepts_X.mean(), accepts_l.mean()

#@njit
def slice_rieman_test(As, theta, n_iter,ls, init=None, prop_X=1, setting="gaussian",m=1):
    """
    Metropolis within Gibbs sampler for the base model.
    - setting can be set to "binary" to handle binary networks
    - prop_X and prop_l are the proposal variances for X and l
    The function returns the final values of X and l, as well as the running likelihood
    and the chain acceptance rates.
    """
    F, mu, sigma, sigma_l = theta
    n_samples = As.shape[0]
    accepts_X = np.zeros((n_iter, n_samples))
    
    n, p = F.shape[-2:]
    X_history=[]
    if init==None:
        mode = st.proj_V(F)
        Xs = np.zeros((n_samples, n, p))
        
        for i in range(n_samples):
            Xs[i] = mode
            
    else:
        Xs = init
        Xs = Xs.copy()
        
    w_current=prop_X
    if setting=="gaussian":
        current_log_lk = np.array([model.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    elif setting=="binary":
        current_log_lk = np.array([model_bin.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    
    for t in range(n_iter):
        X_history.append(Xs.copy())
        for i in range(n_samples):
            if setting=="gaussian":
                cc_log_lk = model.log_lk_partial(Xs[i], ls[i], As[i], theta)
            elif setting=="binary":
                cc_log_lk = model_bin.log_lk_partial(Xs[i], ls[i], As[i], theta)
            #sample the level related
            log_level = np.log(np.random.uniform(0, 1))+cc_log_lk
            #sample the direction of the geodesics
            
            A,Q,R,v_l,e,v_r = sampler_slice_rieman.sample_tangent_sphere(Xs[i])
            
            #looking for an interval of time [a,b] where the geodesics intersects the level set
            a,b = sampler_slice_rieman.stepping_out(w_current,Xs[i],A,Q,R,v_l,e,v_r,param=(ls[i], As[i], theta), log_level=log_level, setting=setting, m=m)
            # looking for a point in the level set
            X2,count = sampler_slice_rieman.shrinkage(a,b,Xs[i],A,Q,R,v_l,e,v_r,param=(ls[i], As[i], theta), log_level=log_level, setting=setting  )
            Xs[i] = X2 # always accepted
            
            
            if setting=="gaussian":
                new_log_lk = model.log_lk_partial(X2, ls[i], As[i], theta)
            elif setting=="binary":
                new_log_lk = model_bin.log_lk_partial(X2, ls[i], As[i], theta)
            #update the log_lk
            current_log_lk[i] = new_log_lk
            #shall we use that for w ? No clear mechanism
            accepts_X[t,i] = count
            #accepts_X[t,i] = 0# a changer ou supprimer
            
            
    return X_history,Xs, accepts_X.mean()



@njit
def slice_rieman(As, theta, n_iter, init=None, prop_X=1, prop_l=0.5, setting="gaussian",m=1):
    """
    Metropolis within Gibbs sampler for the base model.
    - setting can be set to "binary" to handle binary networks
    - prop_X and prop_l are the proposal variances for X and l
    The function returns the final values of X and l, as well as the running likelihood
    and the chain acceptance rates.
    """
    F, mu, sigma, sigma_l = theta
    n_samples = As.shape[0]
    accepts_X = np.zeros((n_iter, n_samples))
    accepts_l = np.zeros((n_iter, n_samples))
    n, p = F.shape[-2:]
    X_history=[]
    if init==None:
        mode = st.proj_V(F)
        Xs = np.zeros((n_samples, n, p))
        ls = sigma_l*np.random.randn(n_samples, p)
        for i in range(n_samples):
            Xs[i] = mode
            ls[i] += mu
    else:
        Xs, ls = init
        Xs = Xs.copy()
        ls = ls.copy()
    w_current=prop_X
    if setting=="gaussian":
        current_log_lk = np.array([model.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    elif setting=="binary":
        current_log_lk = np.array([model_bin.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    
    for t in range(n_iter):
        X_history.append(Xs.copy())
        for i in range(n_samples):
            if setting=="gaussian":
                cc_log_lk = model.log_lk_partial(Xs[i], ls[i], As[i], theta)
            elif setting=="binary":
                cc_log_lk = model_bin.log_lk_partial(Xs[i], ls[i], As[i], theta)
            #sample the level related
            log_level = np.log(np.random.uniform(0, 1))+cc_log_lk
            #sample the direction of the geodesics
            
            A,Q,R,v_l,e,v_r = sampler_slice_rieman.sample_tangent_sphere(Xs[i])
            
            #looking for an interval of time [a,b] where the geodesics intersects the level set
            a,b = sampler_slice_rieman.stepping_out(w_current,Xs[i],A,Q,R,v_l,e,v_r,param=(ls[i], As[i], theta), log_level=log_level, setting=setting, m=m)
            # looking for a point in the level set
            X2,count = sampler_slice_rieman.shrinkage(a,b,Xs[i],A,Q,R,v_l,e,v_r,param=(ls[i], As[i], theta), log_level=log_level, setting=setting  )
            Xs[i] = X2 # always accepted
            
            
            if setting=="gaussian":
                new_log_lk = model.log_lk_partial(X2, ls[i], As[i], theta)
            elif setting=="binary":
                new_log_lk = model_bin.log_lk_partial(X2, ls[i], As[i], theta)
            #update the log_lk
            current_log_lk[i] = new_log_lk
            #shall we use that for w ? No clear mechanism
            accepts_X[t,i] = count
            #accepts_X[t,i] = 0# a changer ou supprimer
            
            # [l] Generate next move
            l2 = ls[i] + prop_l*np.random.randn(p)
            # [l] Compute the acceptance log-probability
            if setting=="gaussian":
                new_log_lk = model.log_lk_partial(Xs[i], l2, As[i], theta)
            elif setting=="binary":
                new_log_lk = model_bin.log_lk_partial(Xs[i], l2, As[i], theta)
            log_alpha = new_log_lk - current_log_lk[i]
            # [l] Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                ls[i] = l2
                current_log_lk[i] = new_log_lk
                accepts_l[t,i] = 1
            else:
                accepts_l[t,i] = 0
            
    return X_history,Xs, ls, current_log_lk.sum(), accepts_X.mean(), accepts_l.mean()

#@njit
def slice_rieman_adapt(As, theta, n_iter, init=None, prop_X=1, prop_l=0.5, setting="gaussian",m=1,use_adapt=False):
    
    """
    Metropolis within Gibbs sampler for the base model.
    - setting can be set to "binary" to handle binary networks
    - prop_X and prop_l are the proposal variances for X and l
    The function returns the final values of X and l, as well as the running likelihood
    and the chain acceptance rates.
    """
    F, mu, sigma, sigma_l = theta
    n_samples = As.shape[0]
    accepts_X = np.zeros((20, n_samples))
    accepts_l = np.zeros((20, n_samples))
    n, p = F.shape[-2:]
    optimal_rate=0.234
    X_history=[]
    if init==None:
        mode = st.proj_V(F)
        Xs = np.zeros((n_samples, n, p))
        ls = sigma_l*np.random.randn(n_samples, p)
        for i in range(n_samples):
            Xs[i] = mode
            ls[i] += mu
    else:
        Xs, ls = init
        Xs = Xs.copy()
        ls = ls.copy()
    w_current=prop_X
    if setting=="gaussian":
        current_log_lk = np.array([model.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    elif setting=="binary":
        current_log_lk = np.array([model_bin.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    
    
    for t in range(n_iter//20):
        for j in range(20):
            X_history.append(Xs)
            for i in range(n_samples):
                if setting=="gaussian":
                    cc_log_lk = model.log_lk_partial(Xs[i], ls[i], As[i], theta)
                elif setting=="binary":
                    cc_log_lk = model_bin.log_lk_partial(Xs[i], ls[i], As[i], theta)
                #sample the level related
                log_level = np.log(np.random.uniform(0, 1))+cc_log_lk
                #sample the direction of the geodesics
                
                A,Q,R,v_l,e,v_r = sampler_slice_rieman.sample_tangent_sphere(Xs[i])
                
                #looking for an interval of time [a,b] where the geodesics intersects the level set
                a,b = sampler_slice_rieman.stepping_out(w_current,Xs[i],A,Q,R,v_l,e,v_r,param=(ls[i], As[i], theta), log_level=log_level, setting=setting, m=m)
                # looking for a point in the level set
                X2,count = sampler_slice_rieman.shrinkage(a,b,Xs[i],A,Q,R,v_l,e,v_r,param=(ls[i], As[i], theta), log_level=log_level, setting=setting  )
                Xs[i] = X2 # always accepted
                
                
                if setting=="gaussian":
                    new_log_lk = model.log_lk_partial(X2, ls[i], As[i], theta)
                elif setting=="binary":
                    new_log_lk = model_bin.log_lk_partial(X2, ls[i], As[i], theta)
                #update the log_lk
                current_log_lk[i] = new_log_lk
                #shall we use that for w ? No clear mechanism
                accepts_X[j,i] = 1/count
                #accepts_X[t,i] = 0# a changer ou supprimer
                
                # [l] Generate next move
                l2 = ls[i] + prop_l*np.random.randn(p)
                # [l] Compute the acceptance log-probability
                if setting=="gaussian":
                    new_log_lk = model.log_lk_partial(Xs[i], l2, As[i], theta)
                elif setting=="binary":
                    new_log_lk = model_bin.log_lk_partial(Xs[i], l2, As[i], theta)
                log_alpha = new_log_lk - current_log_lk[i]
                # [l] Accept or reject
                if np.log(np.random.rand()) < log_alpha:
                    ls[i] = l2
                    current_log_lk[i] = new_log_lk
                    accepts_l[j,i] = 1
                else:
                    accepts_l[j,i] = 0
        rate_X=np.mean(accepts_X)
        if use_adapt:
            adaptive_X = 2*(rate_X > optimal_rate)-1
            w_current = np.exp(np.log(w_current) + 0.5*adaptive_X/(1+i)**0.6)
    print("w",w_current) 
    print("rate_X",rate_X)       
    return X_history,Xs, ls, current_log_lk.sum(), accepts_X.mean(), accepts_l.mean()
#@njit
def mh_adapt_test(As, theta, n_iter,ls, init=None, prop_X=0.01, setting="gaussian"):
    """
    Metropolis within Gibbs sampler for the base model.
    - setting can be set to "binary" to handle binary networks
    - prop_X and prop_l are the proposal variances for X and l
    The function returns the final values of X and l, as well as the running likelihood
    and the chain acceptance rates.
    """
    
    F, mu, sigma, sigma_l = theta
    optimal_rate=0.234
    n_samples = As.shape[0]
    accepts_X = np.zeros((20, n_samples))
   
    n, p = F.shape[-2:]
    if init==None:
        mode = st.proj_V(F)
        Xs = np.zeros((n_samples, n, p))
        
        for i in range(n_samples):
            Xs[i] = mode
            
    else:
        Xs = init
        Xs = Xs.copy()
        
    
    if setting=="gaussian":
        current_log_lk = np.array([model.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    elif setting=="binary":
        current_log_lk = np.array([model_bin.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    X_history=[]
   
    for t in range(n_iter//20):
        for j in range(20):
            X_history.append(Xs.copy())
            for i in range(n_samples):
                # [X] Generate next move
                D = prop_X*np.random.randn(n,p)
                X2 = st.proj_V(Xs[i] + D)
                # [X] Compute the acceptance log-probability
                if setting=="gaussian":
                    new_log_lk = model.log_lk_partial(X2, ls[i], As[i], theta)
                elif setting=="binary":
                    new_log_lk = model_bin.log_lk_partial(X2, ls[i], As[i], theta)
                log_alpha = new_log_lk - current_log_lk[i]
                # [X] Accept or reject
                if np.log(np.random.rand()) < log_alpha:
                    Xs[i] = X2
                    current_log_lk[i] = new_log_lk
                    accepts_X[j,i] = 1
                    
                else:
                    accepts_X[j,i] = 0
                
                
        rate_X=np.mean(accepts_X)
        
        adaptive_X = 2*(rate_X > optimal_rate)-1
        prop_X = np.exp(np.log(prop_X) + 0.5*adaptive_X/(1+n)**0.6)
        
        
    print("prop-X",prop_X)  
    print("rate_X",rate_X)  
    return X_history,Xs, accepts_X.mean()



def mh_adapt(As, theta, n_iter, init=None, prop_X=0.01, prop_l=0.5, setting="gaussian"):
    """
    Metropolis within Gibbs sampler for the base model.
    - setting can be set to "binary" to handle binary networks
    - prop_X and prop_l are the proposal variances for X and l
    The function returns the final values of X and l, as well as the running likelihood
    and the chain acceptance rates.
    """
    
    F, mu, sigma, sigma_l = theta
    optimal_rate=0.234
    n_samples = As.shape[0]
    accepts_X = np.zeros((20, n_samples))
    accepts_l = np.zeros((20, n_samples))
    n, p = F.shape[-2:]
    if init==None:
        mode = st.proj_V(F)
        Xs = np.zeros((n_samples, n, p))
        ls = sigma_l*np.random.randn(n_samples, p)
        for i in range(n_samples):
            Xs[i] = mode
            ls[i] += mu
    else:
        Xs, ls = init
        Xs = Xs.copy()
        ls = ls.copy()
    
    if setting=="gaussian":
        current_log_lk = np.array([model.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    elif setting=="binary":
        current_log_lk = np.array([model_bin.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    X_history=[]
   
    for t in range(n_iter//20):
        for j in range(20):
            X_history.append(Xs)
            for i in range(n_samples):
                # [X] Generate next move
                D = prop_X*np.random.randn(n,p)
                X2 = st.proj_V(Xs[i] + D)
                # [X] Compute the acceptance log-probability
                if setting=="gaussian":
                    new_log_lk = model.log_lk_partial(X2, ls[i], As[i], theta)
                elif setting=="binary":
                    new_log_lk = model_bin.log_lk_partial(X2, ls[i], As[i], theta)
                log_alpha = new_log_lk - current_log_lk[i]
                # [X] Accept or reject
                if np.log(np.random.rand()) < log_alpha:
                    Xs[i] = X2
                    current_log_lk[i] = new_log_lk
                    accepts_X[j,i] = 1
                    
                else:
                    accepts_X[j,i] = 0
                
                # [l] Generate next move
                l2 = ls[i] + prop_l*np.random.randn(p)
                # [l] Compute the acceptance log-probability
                if setting=="gaussian":
                    new_log_lk = model.log_lk_partial(Xs[i], l2, As[i], theta)
                elif setting=="binary":
                    new_log_lk = model_bin.log_lk_partial(Xs[i], l2, As[i], theta)
                log_alpha = new_log_lk - current_log_lk[i]
                # [l] Accept or reject
                if np.log(np.random.rand()) < log_alpha:
                    ls[i] = l2
                    current_log_lk[i] = new_log_lk
                    accepts_l[j,i] = 1
                else:
                    accepts_l[j,i] = 0
        rate_X=np.mean(accepts_X)
        rate_l=np.mean(accepts_l)
        adaptive_X = 2*(rate_X > optimal_rate)-1
        prop_X = np.exp(np.log(prop_X) + 0.5*adaptive_X/(1+n)**0.6)
        
        adaptive_l = 2*(rate_l > optimal_rate)-1
        prop_l = np.exp(np.log(prop_l) + 0.5*adaptive_l/(1+n)**0.6)
    print("prop-X",prop_X)  
    print("rate_X",rate_X)  
    return X_history,Xs, ls, current_log_lk.sum(), accepts_X.mean(), accepts_l.mean()
@njit
def mh(As, theta, n_iter, init=None, prop_X=0.01, prop_l=0.5, setting="gaussian"):
    """
    Metropolis within Gibbs sampler for the base model.
    - setting can be set to "binary" to handle binary networks
    - prop_X and prop_l are the proposal variances for X and l
    The function returns the final values of X and l, as well as the running likelihood
    and the chain acceptance rates.
    """
    F, mu, sigma, sigma_l = theta
    n_samples = As.shape[0]
    accepts_X = np.zeros((n_iter, n_samples))
    accepts_l = np.zeros((n_iter, n_samples))
    n, p = F.shape[-2:]
    if init==None:
        mode = st.proj_V(F)
        Xs = np.zeros((n_samples, n, p))
        ls = sigma_l*np.random.randn(n_samples, p)
        for i in range(n_samples):
            Xs[i] = mode
            ls[i] += mu
    else:
        Xs, ls = init
        Xs = Xs.copy()
        ls = ls.copy()
    X_history=[]
    if setting=="gaussian":
        current_log_lk = np.array([model.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    elif setting=="binary":
        current_log_lk = np.array([model_bin.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    
    for t in range(n_iter):
        X_history.append(Xs[0].copy())
        for i in range(n_samples):
            # [X] Generate next move
            D = prop_X*np.random.randn(n,p)
            X2 = st.proj_V(Xs[i] + D)
            # [X] Compute the acceptance log-probability
            if setting=="gaussian":
                new_log_lk = model.log_lk_partial(X2, ls[i], As[i], theta)
            elif setting=="binary":
                new_log_lk = model_bin.log_lk_partial(X2, ls[i], As[i], theta)
            log_alpha = new_log_lk - current_log_lk[i]
            # [X] Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                Xs[i] = X2
                current_log_lk[i] = new_log_lk
                accepts_X[t,i] = 1
            else:
                accepts_X[t,i] = 0
            
            # [l] Generate next move
            l2 = ls[i] + prop_l*np.random.randn(p)
            # [l] Compute the acceptance log-probability
            if setting=="gaussian":
                new_log_lk = model.log_lk_partial(Xs[i], l2, As[i], theta)
            elif setting=="binary":
                new_log_lk = model_bin.log_lk_partial(Xs[i], l2, As[i], theta)
            log_alpha = new_log_lk - current_log_lk[i]
            # [l] Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                ls[i] = l2
                current_log_lk[i] = new_log_lk
                accepts_l[t,i] = 1
            else:
                accepts_l[t,i] = 0
            
    return X_history,Xs, ls, current_log_lk.sum(), accepts_X.mean(), accepts_l.mean()


def mala(As, theta, n_iter, init=None, progress=True,
         prop_X=0.01, prop_l=0.5, setting="gaussian"):
    """
    Metropolis Adjusted Langevin Algorithm sampler for the base model.
    - setting can be set to "binary" to handle binary networks
    - prop_X and prop_l are the proposal variances for X and l
    The function returns the final values of X and l, as well as the running likelihood
    and the chain acceptance rates.
    """
    F, mu, sigma, sigma_l = theta
    n_samples = As.shape[0]
    accepts_X = np.zeros((n_iter, n_samples))
    accepts_l = np.zeros((n_iter, n_samples))
    n, p = F.shape[-2:]
    if init is None:
        mode = st.proj_V(F)
        Xs = np.array([mode.copy() for _ in range(n_samples)])
        ls = mu[None,:] + sigma_l*np.random.randn(n_samples, p)
    else:
        Xs, ls = init
        Xs = Xs.copy()
        ls = ls.copy()
    
    if setting=="gaussian":
        current_log_lk = np.array([model.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    elif setting=="binary":
        current_log_lk = np.array([model_bin.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    
    step_X = 0.5*prop_X**2
    step_l = 0.5*prop_l**2
    if setting=="gaussian":
        current_grad_X  = np.array([model.log_lk_partial_grad_X(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    elif setting=="binary":
        current_grad_X  = np.array([model_bin.log_lk_partial_grad_X(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    current_grad_X = np.array([g/norm(g) for g in current_grad_X])
    current_drift_X = np.array([st.proj_V(Xs[i] + step_X*current_grad_X[i]) for i in range(n_samples)])
    if setting=="gaussian":
        current_grad_lambda  = np.array([model.log_lk_partial_grad_lambda(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    elif setting=="binary":
        current_grad_lambda  = np.array([model_bin.log_lk_partial_grad_lambda(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    current_grad_lambda = np.array([g/norm(g) for g in current_grad_lambda])
    
    it = trange(n_iter) if progress else range(n_iter)
    for t in it:
        for i in range(n_samples):
            # [X] Generate next move
            D = prop_X*np.random.randn(n,p)
            grad_X = current_grad_X[i]
            drift_X = current_drift_X[i]
            D += step_X * grad_X
            X2 = st.proj_V(Xs[i] + D)
            if setting=="gaussian":
                grad_X2 = model.log_lk_partial_grad_X(X2, ls[i], As[i], theta)
            elif setting=="binary":
                grad_X2 = model_bin.log_lk_partial_grad_X(X2, ls[i], As[i], theta)
            grad_X2 = grad_X2/norm(grad_X2)
            drift_X2 = st.proj_V(X2 + step_X*grad_X2)
            mala_jump = (-st.discr(Xs[i], drift_X2) + st.discr(X2, drift_X)) / (2*prop_X**2)
            # [X] Compute the acceptance log-probability
            if setting=="gaussian":
                new_log_lk = model.log_lk_partial(X2, ls[i], As[i], theta)
            elif setting=="binary":
                new_log_lk = model_bin.log_lk_partial(X2, ls[i], As[i], theta)
            log_alpha = new_log_lk - current_log_lk[i] + mala_jump
            # [X] Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                Xs[i] = X2
                current_log_lk[i] = new_log_lk
                accepts_X[t,i] = 1
                current_grad_X[i] = grad_X2
                current_drift_X[i] = drift_X2
                if setting=="gaussian":
                    g = model.log_lk_partial_grad_lambda(Xs[i], ls[i], As[i], theta)
                elif setting=="binary":
                    g = model_bin.log_lk_partial_grad_lambda(Xs[i], ls[i], As[i], theta)
                current_grad_lambda[i] = g/norm(g)
            else:
                accepts_X[t,i] = 0
            
            # [l] Generate next move
            l2 = ls[i] + prop_l*np.random.randn(p)
            grad_l = current_grad_lambda[i]
            l2 += step_l * grad_l
            if setting=="gaussian":
                grad_l2 = model.log_lk_partial_grad_lambda(Xs[i], l2, As[i], theta)
            elif setting=="binary":
                grad_l2 = model_bin.log_lk_partial_grad_lambda(Xs[i], l2, As[i], theta)
            grad_l2 = grad_l2/norm(grad_l2)
            mala_jump = (-norm(ls[i]-l2-step_l*grad_l2)**2 + norm(l2-ls[i]-step_l*grad_l)**2) / (2*prop_l**2)
            # [l] Compute the acceptance log-probability
            if setting=="gaussian":
                new_log_lk = model.log_lk_partial(Xs[i], l2, As[i], theta)
            elif setting=="binary":
                new_log_lk = model_bin.log_lk_partial(Xs[i], l2, As[i], theta)
            log_alpha = new_log_lk - current_log_lk[i] + mala_jump
            # [l] Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                ls[i] = l2
                current_log_lk[i] = new_log_lk
                accepts_l[t,i] = 1
                current_grad_lambda[i] = grad_l2
                if setting=="gaussian":
                    g = model.log_lk_partial_grad_X(Xs[i], ls[i], As[i], theta)
                elif setting=="binary":
                    g = model_bin.log_lk_partial_grad_X(Xs[i], ls[i], As[i], theta)
                current_grad_X[i] = g/norm(g)
                current_drift_X[i] = st.proj_V(Xs[i] + step_X*current_grad_X[i])
            else:
                accepts_l[t,i] = 0
            
            
        if progress: it.set_postfix({"log_lk": current_log_lk.sum()})
    if progress: print("Acceptance rates", accepts_X.mean(), accepts_l.mean())
        
    return Xs, ls, current_log_lk.sum(), accepts_X.mean(), accepts_l.mean()


@njit
def logsumexp(a):
    """Numba-compatible implementation of the function scipy.special.logsumexp"""
    res = -np.inf
    for x in a:
        res = np.logaddexp(res, x)
    return res


@njit
def mh_cluster(As, theta, n_iter, init=None, prop_X=0.01, prop_l=0.5, T=1, setting="gaussian"):
    """
    Metropolis within Gibbs sampler for the mixture model.
    - setting can be set to "binary" to handle binary networks
    - prop_X and prop_l are the proposal variances for X and l
    - T is the level of tempering for the z variable (cluster labels)
    The function returns the final values of X, l and z, as well as the running likelihood
    and the chain acceptance rates.
    """
    F, mu, sigma, sigma_l, pi = theta
    K = len(pi)
    n, p = F.shape[1:]
    n_samples = As.shape[0]
    accepts_X = np.zeros((n_iter, n_samples))
    accepts_l = np.zeros((n_iter, n_samples))
    vmf_constants = np.array([spa.log_vmf(F[k]) for k in range(K)])
    
    if init is None:
        mode = [st.proj_V(F[k]) for k in range(K)]
        zs = np.array([np.random.randint(K) for _ in range(n_samples)]).astype(np.int32)
        Xs = np.zeros((n_samples, n, p))
        ls = sigma_l*np.random.randn(n_samples, p)
        for i in range(n_samples):
            Xs[i] = mode[zs[i]]
            ls[i] += mu[zs[i]]
    else:
        Xs, ls, zs = init
        Xs = Xs.copy()
        ls = ls.copy()
        zs = zs.copy()
    
    if setting=="gaussian":
        current_log_lk = np.array([model_cluster.log_lk_partial(Xs[i], ls[i], zs[i], As[i], theta) for i in range(n_samples)])
    elif setting=="binary":
        current_log_lk = np.array([model_bin_cluster.log_lk_partial(Xs[i], ls[i], zs[i], As[i], theta) for i in range(n_samples)])
    
    for t in range(n_iter):
        for i in range(n_samples):
            # [z] Explicit sampling on z
            if setting=="gaussian":
                log_probs = (1/T) * model_cluster.log_lk_partial_z(Xs[i], ls[i], As[i], theta, constants=vmf_constants)
            elif setting=="binary":
                log_probs = (1/T) * model_bin_cluster.log_lk_partial_z(Xs[i], ls[i], As[i], theta, constants=vmf_constants)
            s = logsumexp(log_probs)
            probs = np.exp(log_probs - s)

            # Sample z manually from its cdf (np.random.choice is not availabla in numba)
            cumulative_distribution = np.cumsum(probs)
            cumulative_distribution /= cumulative_distribution[-1]
            u = np.random.rand()
            zs[i] = np.searchsorted(cumulative_distribution, u, side="right")
            
            # [X] Generate next move
            D = prop_X*np.random.randn(n,p)
            X2 = st.proj_V(Xs[i] + D)
            # [X] Compute the acceptance log-probability
            if setting=="gaussian":
                new_log_lk = model_cluster.log_lk_partial(X2, ls[i], zs[i], As[i], theta)
            elif setting=="binary":
                new_log_lk = model_bin_cluster.log_lk_partial(X2, ls[i], zs[i], As[i], theta)
            log_alpha = new_log_lk - current_log_lk[i]
            # [X] Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                Xs[i] = X2
                current_log_lk[i] = new_log_lk
                accepts_X[t,i] = 1
            else:
                accepts_X[t,i] = 0
            
            # [l] Generate next move
            l2 = ls[i] + prop_l*np.random.randn(p)
            # [l] Compute the acceptance log-probability
            if setting=="gaussian":
                new_log_lk = model_cluster.log_lk_partial(Xs[i], l2, zs[i], As[i], theta)
            elif setting=="binary":
                new_log_lk = model_bin_cluster.log_lk_partial(Xs[i], l2, zs[i], As[i], theta)
            log_alpha = new_log_lk - current_log_lk[i]
            # [l] Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                ls[i] = l2
                current_log_lk[i] = new_log_lk
                accepts_l[t,i] = 1
            else:
                accepts_l[t,i] = 0
            
    return Xs, ls, zs, current_log_lk.sum(), accepts_X.mean(), accepts_l.mean()

@njit
def slice_mask(A, mask, theta, n_iter, init=None, progress=True, prop_X=1,m=10):
    """
    Given a set of coefficients of A, runs a MCMC chain to sample from the
    remaining hidden coefficients and the latent variables (X, l).
    - mask is the set of unknown coefficients, given as two arrays of x and y indices
    - prop_X is the initial proposal variance
    The function returns the arrays of values of A, X and l along the MCMC.
    """
    A_init = A.copy()
    F, mu, sigma, sigma_l = theta
    mx, my = mask
    accepts_X = np.zeros(n_iter)
    n, p = F.shape
    batch = 50
    optimal_rate = 0.234
    if init is None:
        X = st.proj_V(F)
        l = mu.copy()
    else:
        A, X, l = init
    
    # Posterior variance for lambda:
    posterior_std_l = np.sqrt(1/(1/sigma**2 + 1/sigma_l**2))
    sv_F = np.array([norm(F[:,i]) for i in range(p)])
    lks = np.zeros(n_iter)
    A_mh = np.zeros((n_iter, n, n))
    X_mh = np.zeros((n_iter, n, p))
    l_mh = np.zeros((n_iter, p))
    
    it = range(n_iter)
    for t in it:
        lks[t] = model.log_lk_partial(X, l, A, theta)
        # Sample on A
        A2 = A_init.copy()
        comp = st.comp_numba_single(X, l)
        for i in range(len(mx)):
            eps = sigma*np.sqrt(2)*np.random.randn()
            A2[mx[i], my[i]] = comp[mx[i],my[i]] + eps
        A = (A2+A2.T)/2
        
        
        current_log_lk  = model.log_lk_partial(X, l, A, theta)
        #sample the level related
        log_level = np.log(np.random.uniform(0, 1))+current_log_lk 
        #sample the direction of the geodesics
        AA,Q,R,v_l,e,v_r = sampler_slice_rieman.sample_tangent_sphere(X)
        
        #looking for an interval of time [a,b] where the geodesics intersects the level set
        a,b = sampler_slice_rieman.stepping_out(prop_X,X,AA,Q,R,v_l,e,v_r,param=(l, A, theta), log_level=log_level, setting="gaussian", m=m)
        # looking for a point in the level set
        X2,count = sampler_slice_rieman.shrinkage(a,b,X,AA,Q,R,v_l,e,v_r,param=(l, A, theta), log_level=log_level, setting="gaussian"  )
        X = X2 # always accepted
        new_log_lk = model.log_lk_partial(X2, l, A, theta)
       
        #update the log_lk
        current_log_lk  = new_log_lk
        accepts_X[t] =1/count

        # Sample on lambda
        v = np.diag(X.T@A@X)
        posterior_mean = (posterior_std_l**2)*(v/sigma**2 + mu/sigma_l**2)
        l = posterior_mean
        
        A_mh[t] = A
        X_mh[t] = X
        l_mh[t] = l
        
        # Adaptively tune the acceptance rate
        # if t%batch==0 and t>1:
        #     rate_X = accepts_X[max(0, t-batch):t+1].mean()
        #     adaptive_X = 2*(rate_X > optimal_rate)-1
        #     prop_X = np.exp(np.log(prop_X) + 0.5*adaptive_X/np.sqrt(1+n))
        
    return A_mh, X_mh, l_mh, lks
@njit
def mh_mask(A, mask, theta, n_iter, init=None, progress=True, prop_X=0.02):
    """
    Given a set of coefficients of A, runs a MCMC chain to sample from the
    remaining hidden coefficients and the latent variables (X, l).
    - mask is the set of unknown coefficients, given as two arrays of x and y indices
    - prop_X is the initial proposal variance
    The function returns the arrays of values of A, X and l along the MCMC.
    """
    A_init = A.copy()
    F, mu, sigma, sigma_l = theta
    mx, my = mask
    accepts_X = np.zeros(n_iter)
    n, p = F.shape
    batch = 50
    optimal_rate = 0.234
    if init is None:
        X = st.proj_V(F)
        l = mu.copy()
    else:
        A, X, l = init
    
    # Posterior variance for lambda:
    posterior_std_l = np.sqrt(1/(1/sigma**2 + 1/sigma_l**2))
    sv_F = np.array([norm(F[:,i]) for i in range(p)])
    lks = np.zeros(n_iter)
    A_mh = np.zeros((n_iter, n, n))
    X_mh = np.zeros((n_iter, n, p))
    l_mh = np.zeros((n_iter, p))
    
    it = range(n_iter)
    for t in it:
        lks[t] = model.log_lk_partial(X, l, A, theta)
        # Sample on A
        A2 = A_init.copy()
        comp = st.comp_numba_single(X, l)
        for i in range(len(mx)):
            eps = sigma*np.sqrt(2)*np.random.randn()
            A2[mx[i], my[i]] = comp[mx[i],my[i]] + eps
        A = (A2+A2.T)/2
        
        # [X] Generate next move
        D = prop_X*np.random.randn(n,p)/sv_F
        X2 = st.proj_V(X + D)
        # [X] Compute the acceptance log-probability
        current_log_lk = model.log_lk_partial(X, l, A, theta)
        new_log_lk = model.log_lk_partial(X2, l, A, theta)
        log_alpha = (new_log_lk - current_log_lk) * 100
        # [X] Accept or reject
        if np.log(np.random.rand()) < log_alpha:
            X = X2
            current_log_lk = new_log_lk
            accepts_X[t] = 1
        else:
            accepts_X[t] = 0

        # Sample on lambda
        v = np.diag(X.T@A@X)
        posterior_mean = (posterior_std_l**2)*(v/sigma**2 + mu/sigma_l**2)
        l = posterior_mean
        
        A_mh[t] = A
        X_mh[t] = X
        l_mh[t] = l
        
        # Adaptively tune the acceptance rate
        if t%batch==0 and t>1:
            rate_X = accepts_X[max(0, t-batch):t+1].mean()
            adaptive_X = 2*(rate_X > optimal_rate)-1
            prop_X = np.exp(np.log(prop_X) + 0.5*adaptive_X/np.sqrt(1+n))
        
    return A_mh, X_mh, l_mh, lks


@njit
def map_mask(A, mask, theta, n_iter):
    """
    Given a set of coefficients of A, finds the MAP estimator of the
    remaining hidden coefficients and the latent variables (X, l).
    mask is the set of unknown coefficients, given as two arrays of x and y indices.
    The function returns the arrays of values of A, X and l along the MCMC.
    """
    F, mu, sigma, sigma_l = theta
    mx, my = mask
    accepts_X = np.zeros(n_iter)
    n, p = F.shape
    batch = 50

    A = A.copy()
    X = st.proj_V(F)
    l = mu.copy()
    # Posterior standard deviation for lambda:
    posterior_std_l = np.sqrt(1/(1/sigma**2 + 1/sigma_l**2))
    lks = np.zeros(n_iter)
    
    it = range(n_iter)
    for t in it:
        step = 1/(2*t+1)
        
        # [A] Explicit maximum on A
        comp = st.comp_numba_single(X, l)
        for i in range(len(mx)):
            A[mx[i], my[i]] = comp[mx[i],my[i]]
            A[my[i], mx[i]] = comp[mx[i],my[i]]
        
        # [X] Sample on X
        grad_X = model.log_lk_partial_grad_X(X, l, A, theta)
        grad_X = grad_X/norm(grad_X)
        X = st.proj_V(X + step*grad_X)

        # [l] Explicit maximum on lambda
        v = np.diag(X.T@A@X)
        l = (posterior_std_l**2)*(v/sigma**2 + mu/sigma_l**2)
        
        lks[t] = model.log_lk_partial(X, l, A, theta)
        
    return A, X, l, lks