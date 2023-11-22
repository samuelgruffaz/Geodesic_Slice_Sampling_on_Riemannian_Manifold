"""
This file contains the same functions as in model_cluster.py, adapted for the case where
the adjacency matrices have binary coefficients. The matrix coefficients are then modeled
as independent Bernoulli distributions.
"""


import numpy as np
from numba import njit

import src.stiefel as st
from src import spa
from src.model_bin import sigmoid


def log_lk(Xs, ls, zs, As, theta, normalized=False):
    """
    Log-density [Xs, ls, zs, As | theta]
    """
    n_samples, n_nodes, _ = As.shape
    F, mu, sigma, sigma_l, pi = theta
    res = 0
    
    # (As | Xs, ls)
    prob = sigmoid(st.comp_numba_many(Xs, ls))
    res += (As*np.log(prob) + (1-As)*np.log(1-prob)).sum()
    
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
    n_nodes, _ = A.shape
    F, mu, sigma, sigma_l, pi = theta
    res = 0
    
    # (As | Xs, ls)
    prob = sigmoid(st.comp_numba_single(X, l))
    res += (A*np.log(prob) + (1-A)*np.log(1-prob)).sum()
    
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
    n_nodes, _ = A.shape
    F, mu, sigma, sigma_l, pi = theta
    K = len(F)
    log_prob = np.zeros(K)
    prob = sigmoid(st.comp_numba_single(X, l))
    A_X = (A*np.log(prob) + (1-A)*np.log(1-prob)).sum()
    
    for k in range(K):
        # (As | Xs)
        res = A_X

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
    grad = (mu[z]-l)/sigma_l[z]**2
    prob = sigmoid(st.comp_numba_single(X, l))
    C = A*(1-prob) - (1-A)*prob
    for i in range(n):
        for j in range(n):
            grad += C[i,j]*X[i]*X[j]
    return grad

@njit
def log_lk_partial_grad_X(X, l, z, A, theta):
    """Riemannian gradient of log_lk_partial with respect to X"""
    F, mu, sigma, sigma_l, pi = theta
    grad_E = F[z]
    prob = sigmoid(st.comp_numba_single(X, l))
    C = A*(1-prob) - (1-A)*prob
    for i in range(n):
        for j in range(n):
            if i==j:
                grad_E[i] += 2*C[i,i]*l*X[j]
            else:
                grad_E[i] += C[i,j]*l*X[j]
                grad_E[j] += C[i,j]*l*X[i]
    grad_R = (grad_E@X.T - X@grad_E.T)@X # Riemannian gradient for the canonical metric
    return grad_R