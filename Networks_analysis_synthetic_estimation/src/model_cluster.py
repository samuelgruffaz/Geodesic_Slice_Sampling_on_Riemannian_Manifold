"""
This file contains functions to compute the mixture model's full and partial log-densities.
The variables are named as follows:
- As in the code corresonds to (A_1, ..., A_N) in the paper
- Xs in the code corresonds to (X_1, ..., X_N) in the paper
- ls in the code corresonds to (lambda_1, ..., lambda_N) in the paper
- zs in the code corresonds to (z_1, ..., z_N) in the paper

The partial log-likelihood functions, which are used extensively in the MCMC-SAEM, are compiled with Numba. 
In some function, including the normalizing constant is optional, as it is the most time-intensive step.

The parameter theta is composed of:
- F and mu designate the list of vMF parameters and the mean eigenvalues for each cluster.
- sigma and sigma_l in the code are the list of values of sigma_epsilon and sigma_lambda for each cluster.
"""

import numpy as np
from numba import njit

import src.stiefel as st
from src import spa


def log_lk(Xs, ls, zs, As, theta, normalized=False):
    """
    Log-density [Xs, ls, zs, As | theta]
    """
    n_samples, n, _ = As.shape
    F, mu, sigma, sigma_l, pi = theta
    res = 0
    
    # (As | Xs, ls, sigma)
    Xs_comp = st.comp(Xs, ls)
    res += -0.5*(((As-Xs_comp)**2)/(sigma[zs][:,None,None]**2)).sum() - (n**2)*np.log(sigma[zs]).sum()

    # (Xs | F)
    res += (F[zs]*Xs).sum()
    if normalized:
        constants = np.array([spa.log_vmf(F[k]) for k in range(len(F))])
        res += -constants[zs].sum()

    # (ls | mu, sigma_l)
    p = F.shape[1]
    res += -0.5*(((ls-mu[zs])**2)/(sigma_l[zs][:,None]**2)).sum() - p*np.log(sigma_l[zs]).sum()

    # (zs)
    res += np.log(pi[zs]).sum()
    
    return res


@njit
def log_lk_partial(X, l, z, A, theta, normalized=False):
    """
    Log-density [Xs[i], ls[i], zs[i], As[i] | theta]: log-likelihood term for one individual
    """
    n, _ = A.shape
    F, mu, sigma, sigma_l, pi = theta
    res = 0
    # (As | Xs)
    X_comp = st.comp_numba_single(X, l)
    As_Xs = -0.5*((A-X_comp)**2).sum()/sigma[z]**2 - (n**2)*np.log(sigma[z])
    res += As_Xs

    # (Xs | F)
    res += (F[z]*X).sum()
    if normalized:
        res += -spa.log_vmf(F[z])
        
    # (ls | mu, sigma_l)
    p = F.shape[1]
    res += -0.5*((l-mu[z])**2).sum()/sigma_l[z]**2 - p*np.log(sigma_l[z])
    
    # (zs)
    res += np.log(pi[z])
    
    return res

@njit
def log_lk_partial_z(X, l, A, theta, constants=None):
    """
    This function gives the log-probabilities of each cluster given the observed variables (X, l, A)
    The argument `constants` is the list of logarithmic normalizing constants of every vMF parameter in F.
    """
    n, _ = A.shape
    F, mu, sigma, sigma_l, pi = theta
    K = len(F)
    log_prob = np.zeros(K)
    X_comp = st.comp_numba_single(X, l)
    A_X_comp = ((A-X_comp)**2).sum()
    
    for k in range(K):
        res = 0
        # (As | Xs)
        As_Xs = -0.5*A_X_comp/sigma[k]**2 - (n**2)*np.log(sigma[k])
        res += As_Xs

        # (Xs | F)
        res += (F[k]*X).sum()
        res += -constants[k]

        # (ls | mu, sigma_l)
        p = F.shape[1]
        res += -0.5*((l-mu[k])**2).sum()/sigma_l[k]**2 - p*np.log(sigma_l[k])

        # (zs)
        res += np.log(pi[k])
        log_prob[k] = res
    return log_prob

@njit
def log_lk_partial_grad_lambda(X, l, z, A, theta):
    """Gradient of log_lk_partial with respect to lambda"""
    F, mu, sigma, sigma_l, pi = theta
    n, p = X.shape
    grad = -(1/sigma[z]**2 + 1/sigma_l[z]**2) * l
    tmp = np.zeros(p)
    for k in range(p):
        for i in range(n):
            for j in range(n):
                tmp[k] += X[i,k] * A[i,j] * X[j,k]
    grad += tmp / sigma[z]**2
    grad += mu / sigma_l[z]**2
    return grad

@njit
def log_lk_partial_grad_X(X, l, z, A, theta):
    """Riemannian gradient of log_lk_partial with respect to X"""
    F, mu, sigma, sigma_l, pi = theta
    grad_E = A@X@np.diag(l) / sigma[z]**2 + F
    grad_R = (grad_E@X.T - X@grad_E.T)@X # Riemannian gradient for the canonical metric
    return grad_R