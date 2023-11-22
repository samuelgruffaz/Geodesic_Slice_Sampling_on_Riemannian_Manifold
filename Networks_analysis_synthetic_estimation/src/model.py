"""
This file contains functions to compute the model's full and partial log-densities, as well as their gradients.
The variables are named as follows:
- As in the code corresonds to (A_1, ..., A_N) in the paper
- Xs in the code corresonds to (X_1, ..., X_N) in the paper
- ls in the code corresonds to (lambda_1, ..., lambda_N) in the paper

The partial log-likelihood functions, which are used extensively in the MCMC-SAEM, are compiled with Numba.
In some function, including the normalizing constant is optional, as it is the most time-intensive step.

The parameter theta is composed of:
- F and mu designate the vMF parameter and the mean eigenvalues
- sigma and sigma_l in the code correspond to sigma_epsilon and sigma_lambda in the paper
"""

import numpy as np
from numba import njit
from src.Sampler_class import *

import src.stiefel as st
from src import spa, vmf


def log_lk(Xs, ls, As, theta, normalized=False):
    """
    Log-density [Xs, ls, As | theta]
    """
    n_samples, n_nodes, _ = As.shape
    F, mu, sigma, sigma_l = theta
    res = 0
    
    # (As | Xs, ls, sigma)
    Xs_comp = st.comp(Xs, ls)
    res += -0.5*((As-Xs_comp)**2).sum()/sigma**2 - n_samples*(n_nodes**2)*np.log(sigma)
    
    # (Xs | F)
    res += (F*Xs).sum()
    if normalized:
        res += -n_samples*spa.log_vmf(F)
    
    # (ls | mu, sigma_l)
    p = F.shape[1]
    res += -0.5*((ls-mu[None,:])**2).sum()/sigma_l**2 - n_samples*p*np.log(sigma_l)
    
    return res

def post_log_X(X, A, theta):
    """
    Log-density [X  |A, theta] without normalizing consants
    """
    
    F, mu, sigma, sigma_l = theta
    res = 0
    
    # (X | A,theta)
    sigma_p_invsquarre=1/sigma**2+1/sigma_l**2
    mux=(mu/sigma_l**2+st.Ax_numba(A,X)/sigma**2)/sigma_p_invsquarre
    norm_mux_squarre=np.linalg.norm(mux)**2
    res += sigma_p_invsquarre*norm_mux_squarre/2
    
    # (X | F)
    res += (F*X).sum()
    
    return res



def IS_ls_samples(As,Xs_samp,theta,n_ss=100):
    # sample from a normal distribution related to the IS sampling
    #Xs_samp(n_samples,n_ss,n,p)
    F, mu, sigma, sigma_l = theta
    n,p=F.shape
    n_samples=len(As)
    sigma_p_invsquarre=1/sigma**2+1/sigma_l**2
    lss=np.zeros((n_samples,n_ss,p))
    muxs=np.zeros((n_samples,n_ss,p))
    F, mu, sigma, sigma_l = theta
    gauss_standard=np.random.standard_normal(size=(n_samples,n_ss,p))
    for i in range(n_samples):
        for j in range(n_ss):
            mux=(mu/sigma_l**2+st.Ax_numba(As[i],Xs_samp[i,j])/sigma**2)/sigma_p_invsquarre
            muxs[i,j,:]=mux.copy()
            lss[i,j,:]=mux.copy()+gauss_standard[i,j,:]/np.sqrt(sigma_p_invsquarre)

    return lss,muxs#(N_samples,n_ss,p )

def IS_Xs_samples(F_chaps,n_ss=100,stride=10,burn=500):
    """
    Samples from the IS distribution (X|F_chaps)
    F_chaps (n_samples,n,p)
    """
    
    n,p=F_chaps[0].shape
    n_samples=len(F_chaps)
    Xss=np.zeros((n_samples,n_ss,n,p))
    total_steps=stride*n_ss+burn
    Ind_sel=[ s for s in range(burn,total_steps,stride)]
    for i in range(n_samples):
        X_0=st.proj_V(F_chaps[i])
        log_prob = lambda x : (F_chaps[i]*x).sum()
        sampler_slice=Sampler_Slice_Sampling_Geodesics(log_prob,n,p)
        
        data = sampler_slice.run_kernel(X_0, total_steps,w=1,use_adapt=False,m=10)
    
        for k,j in enumerate(Ind_sel):
            Xss[i,k,:,:]=data[j]
    return Xss
def samples_posterior_z(As,theta,Init=None,n_ss=100,stride=10,burn=500):
    F, mu, sigma, sigma_l = theta
    n,p=F.shape
    n_samples=len(As)
    if Init is None:
        Init=st.proj_V(F)+np.zeros((n_samples,n,p))
    total_steps=stride*n_ss+burn
    Ind_sel=[ s for s in range(burn,total_steps,stride)]
    X_tofit_s=np.zeros((n_samples,len(Ind_sel),n,p))
    for i in range(n_samples):
        
        log_prob = lambda x : post_log_X(x, As[i], theta)
        sampler_slice=Sampler_Slice_Sampling_Geodesics(log_prob,n,p)
        
        data = sampler_slice.run_kernel(Init[i], total_steps,w=1,use_adapt=False,m=10)
    
        for k,j in enumerate(Ind_sel):
            X_tofit_s[i,k,:,:]=data[j]
    return X_tofit_s

def generate_IS_param(As,theta,Init=None,n_ss=100,stride=10,burn=500):
    F, mu, sigma, sigma_l = theta
    n,p=F.shape
    n_samples=len(As)
    F_chaps=np.zeros((n_samples,n,p))
    X_tofit_s=samples_posterior_z(As,theta,Init=Init,n_ss=n_ss,stride=stride,burn=burn)
    # Update F
    for i in range(n_samples):
        F_chaps[i]= vmf.mle(X_tofit_s[i].mean(axis=0), orth=True)
    
    return F_chaps
def IS_lk(As,theta,Init=None,n_ss=100,stride=10,burn=500):
    F_chaps=generate_IS_param(As,theta,Init=Init,n_ss=n_ss,stride=stride,burn=burn)
    Xss=IS_Xs_samples(F_chaps,n_ss=n_ss,stride=stride,burn=burn)
    lss,muxs=IS_ls_samples(As,Xss,theta,n_ss=n_ss)
    
    lk=log_ratio(Xss[:,0,:,:],lss[:,0,:],As,F_chaps,muxs[:,0,:],theta)
    for i in range(1,n_ss):
        lk=np.logaddexp(lk,log_ratio(Xss[:,i,:,:],lss[:,i,:],As,F_chaps,muxs[:,i,:],theta))
    return lk-np.log(n_ss)

def bridge_log_ratio_num(log_last_guess,Xs,ls,As,F_chaps,muxs,theta):
    l_2i=log_ratio(Xs,ls,As,F_chaps,muxs,theta)
    l_2prime=np.logaddexp(l_2i,log_last_guess)

    return l_2i-l_2prime

def bridge_log_ratio_den(log_last_guess,Xs,ls,As,F_chaps,muxs,theta):
    l_1i=log_ratio(Xs,ls,As,F_chaps,muxs,theta)
    l_1prime=np.logaddexp(l_1i,log_last_guess)

    return -l_1prime


def bridge_step_estimate(log_last_guess,As,theta,F_chaps,samples):
    #elementary step of the bridge sampling estimator
    Xss,lss,muxs,Xss_post,lss_post,muxs_post=samples
    n_ss=Xss.shape[1]

    lk_num=bridge_log_ratio_num(log_last_guess,Xss[:,0,:,:],lss[:,0,:],As,F_chaps,muxs[:,0,:],theta)
    lk_den=bridge_log_ratio_den(log_last_guess,Xss_post[:,0,:,:],lss_post[:,0,:],As,F_chaps,muxs_post[:,0,:],theta)
    #N_1=N_2 here
    for i in range(1,n_ss):
        lk_num=np.logaddexp(lk_num,bridge_log_ratio_num(log_last_guess,Xss[:,i,:,:],lss[:,i,:],As,F_chaps,muxs[:,i,:],theta))
        lk_den=np.logaddexp(lk_den,bridge_log_ratio_den(log_last_guess,Xss_post[:,i,:,:],lss_post[:,i,:],As,F_chaps,muxs_post[:,i,:],theta))

    return lk_num-lk_den

def criteria_go(log_last,log_curent,k):# à affiner
    val=np.logaddexp(log_last-log_curent,log_curent-log_last)#|a-b|^2/|b||a|
    if val>np.log(2)+1:
        return (k<10)
    else:
        return (k<10)*(np.sqrt(np.exp(val)-2)>10**(-10))

def Bridge_sampling_lk(As,theta,Init=None,n_ss_l=[100,1000],stride_l=[10,5],burn_l=[1000,1000]):
    # Bridge sampling to estimate p(As|theta), Init should be (n_samples,n,p) to initialize Xs
    F_chaps=generate_IS_param(As,theta,Init=Init,n_ss=n_ss_l[0],stride=stride_l[0],burn=burn_l[0]) # We estimate parameter of the proposal distribution
    Xss=IS_Xs_samples(F_chaps,n_ss=n_ss_l[1],stride=stride_l[1],burn=burn_l[1])# samples of the proposal
    lss,muxs=IS_ls_samples(As,Xss,theta,n_ss=n_ss_l[1])# samples of the proposal, mux is an intermediary computation
    #lk_num=log_ratio(Xss[:,0,:,:],lss[:,0,:],As,F_chaps,muxs[:,0,:],theta)

    #using posterior samples generated in generate_IS_param will bias the estimate
    Xss_post=samples_posterior_z(As,theta,Init=Init,n_ss=n_ss_l[1],stride=stride_l[1],burn=burn_l[1])#samples of the posterior
    lss_post,muxs_post=IS_ls_samples(As,Xss_post,theta,n_ss=n_ss_l[1])#samples of the posterior
    samples=(Xss,lss,muxs,Xss_post,lss_post,muxs_post)
    log_last_guess=-200# near 0
    log_current_guess=bridge_step_estimate(log_last_guess,As,theta,F_chaps,samples)
    k=0
    while criteria_go(log_last_guess,log_current_guess,k):
        print(log_current_guess)
        log_current_guess,log_last_guess=bridge_step_estimate(log_current_guess,As,theta,F_chaps,samples),log_current_guess
        k=k+1
    return log_current_guess,(k<10)
        

def log_ratio(Xs,ls,As,F_chaps,muxs,theta):

    #log p(X,l,A|theta)/IS(X,l)

    
    F, mu, sigma, sigma_l = theta
    n,p=F.shape
    sigma_p_inv_squarre=1/sigma**2+1/sigma_l**2
    n_samples=len(Xs)
    den=0
    #log IS(X,l)
    for i in range(n_samples):

        den+=sigma_p_inv_squarre*np.linalg.norm(ls-muxs[i])/2+p*(np.log(sigma_p_inv_squarre)-np.log(2*np.pi))/2# l|X,theta,A
        den+=-spa.log_vmf(F_chaps[i])+(F_chaps[i]*Xs[i]).sum()# (X|theta,A)_est part

    #p(X,l,A|theta)
    # (As | Xs, ls, sigma)
    Xs_comp = st.comp(Xs, ls)
    nom= -0.5*((As-Xs_comp)**2).sum()/sigma**2 - n_samples*(n**2)*np.log(sigma)- n_samples*(n**2)*np.log(2*np.pi)/2
    nom += -n_samples*spa.log_vmf(F)
    
    # (ls | mu, sigma_l)
    p = F.shape[1]
    nom += -0.5*((ls-mu[None,:])**2).sum()/sigma_l**2 - n_samples*p*np.log(sigma_l)- n_samples*p*np.log(2*np.pi)/2
    
    return nom-den



@njit
def log_lk_partial(X, l, A, theta, normalized=False):
    """
    Log-density [Xs[i], ls[i], As[i] | theta]: log-likelihood term for one individual
    """
    n_nodes, _ = A.shape
    F, mu, sigma, sigma_l = theta
    res = 0
    # (As | Xs)
    X_comp = st.comp_numba_single(X, l)
    As_Xs = -0.5*((A-X_comp)**2).sum()/sigma**2 - (n_nodes**2)*np.log(sigma)
    res += As_Xs
    
    # (Xs | F)
    res += (F*X).sum()
    if normalized:
        res += -spa.log_vmf(F)
        
    # (ls | mu, sigma_l)
    p = F.shape[1]
    res += -0.5*((l-mu)**2).sum()/sigma_l**2 - p*np.log(sigma_l)
    
    return res

@njit
def log_lk_partial_grad_lambda(X, l, A, theta):
    """Gradient of log_lk_partial with respect to lambda"""
    F, mu, sigma, sigma_l = theta
    n, p = X.shape
    grad = -(1/sigma**2 + 1/sigma_l**2) * l
    tmp = np.zeros(p)
    for k in range(p):
        for i in range(n):
            for j in range(n):
                tmp[k] += X[i,k] * A[i,j] * X[j,k]
    grad += tmp / sigma**2
    grad += mu / sigma_l**2
    return grad

@njit
def log_lk_partial_grad_X(X, l, A, theta):
    """Riemannian gradient of log_lk_partial with respect to X"""
    F, mu, sigma, sigma_l = theta
    grad_E = A@X@np.diag(l) / sigma**2 + F
    grad_R = (grad_E@X.T - X@grad_E.T)@X # Riemannian gradient for the canonical metric
    return grad_R