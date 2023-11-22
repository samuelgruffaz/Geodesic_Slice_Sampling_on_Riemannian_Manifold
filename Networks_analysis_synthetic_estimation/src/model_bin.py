"""
This file contains the same functions as in model.py, adapted for the case where
the adjacency matrices have binary coefficients. The matrix coefficients are then modeled
as independent Bernoulli distributions.
"""


import numpy as np
from numba import njit

import src.stiefel as st
from src import spa


@njit
def sigmoid(x):
    return 1/(1+np.exp(-x))


def log_lk(Xs, ls, As, theta, normalized=False):
    """
    Log-density [Xs, ls, As | theta]
    """
    n_samples, n_nodes, _ = As.shape
    F, mu, sigma, sigma_l = theta
    res = 0
    
    # (As | Xs, ls)
    prob = sigmoid(st.comp_numba_many(Xs, ls))
    res += (As*np.log(prob) + (1-As)*np.log(1-prob)).sum()
    
    # (Xs | F)
    res += (F*Xs).sum()
    if normalized:
        res += -n_samples*spa.log_vmf(F)
    
    # (ls | mu, sigma_l)
    p = F.shape[1]
    res += -0.5*((ls-mu[None,:])**2).sum()/sigma_l**2 - n_samples*p*np.log(sigma_l)
    
    return res


@njit
def log_lk_partial(X, l, A, theta, normalized=False):
    """
    Log-density [Xs[i], ls[i], As[i] | theta]: log-likelihood term for one individual
    """
    F, mu, sigma, sigma_l = theta
    res = 0
    
    # (As | Xs, ls)
    prob = sigmoid(st.comp_numba_single(X, l))
    res += (A*np.log(prob) + (1-A)*np.log(1-prob)).sum()
    
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
    grad = (mu-l)/sigma_l**2
    prob = sigmoid(st.comp_numba_single(X, l))
    C = A*(1-prob) - (1-A)*prob
    for i in range(n):
        for j in range(n):
            grad += C[i,j]*X[i]*X[j]
    return grad


@njit
def log_lk_partial_grad_X(X, l, A, theta):
    """Riemannian gradient of log_lk_partial with respect to X"""
    F, mu, sigma, sigma_l = theta
    n, p = X.shape
    grad_E = F
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