"""
Python translation of the Matlab implementation of Kume et al. 2013,
which allow computing the Saddle-Point Approximation of the normalizing
constant of the von Mises-Fisher distribution on the Stiefel manifold.
For further details, we refer the reader to the original paper.

The main function in this file is log_vmf(F), which computes the logarithm of the saddle-point approximation.

The functions are compiled with Numba in order to accelerate the code.
"""


import numpy as np
from numba import njit


@njit
def get_r_s(k):
    """
    Suppose that k is the index of the vectorized elements of an upper
    triangular square matrix, e.g. of the form,
       1 2 4 7
       0 3 5 8
       0 0 6 9
       0 0 0 10
    then [r,s] are the corresponding co-ordinates.  
    E.g get_r_s(9) returns [3,4].
    """
    s = np.floor(np.sqrt(2*k - 7/4) + 1/2)
    r = k - 1/2*s*(s-1)
    return int(r)-1, int(s)-1


@njit
def calc_KHat2_matrixFisher(phiHat_d, lambda_d, n):
    p = len(lambda_d)
    leng = int(1/2*p*(p+1))
    out = np.zeros((leng, leng))
    for i in range(leng):
        for j in range(leng):
            r1,s1 = get_r_s(i+1)
            r2,s2 = get_r_s(j+1)
            if r1==r2 and s1==s2:
                if r2==s1:
                    out[i,j] = 2*n*phiHat_d[r1]**2 + 4*(lambda_d[r1]**2)*(phiHat_d[r1]**3);
                else:
                    out[i,j] = n*phiHat_d[r1]*phiHat_d[s1] + \
                        phiHat_d[r1]*phiHat_d[s1]*((lambda_d[r1]**2)*phiHat_d[r1] + \
                        (lambda_d[s1]**2)*phiHat_d[s1]);
    return out


@njit
def compute_T(n, p, s, phi_hat):
    # Add the R_k(T) term
    no_I = np.ones((p,p)) - np.eye(p) # = 1 iff i != j
    h = s**2 * phi_hat / n
    
    # rho_13^2
    
    runsum = 0
    for r in range(p):
        foo = 0
        for s in range(p):
            if r != s:
                foo = foo + (1+2*h[r]+h[s])/(1+h[r]+h[s])
        runsum = runsum + 1/(1+2*h[r])*(2*(1+3*h[r])/(1+2*h[r])+foo)**2
    rho_13_2 = runsum*2/n
    
    # sigma_1
    sigma_1 = ((1+3*h)**2 / (1+2*h)**3).sum()
    
    # sigma_2
    sigma_2 = 0
    for r in range(p):
        for s in range(p):
            if s != r:
                sigma_2 = sigma_2 + (1+2*h[r]+h[s])**2/((1+2*h[r])*(1+h[r]+h[s])**2)
    
    # sigma_3
    sigma_3 = 0
    for r in range(p):
        for s in range(r+1,p):
            for t in range(s+1,p):
                sigma_3 = sigma_3 + (1+h[r]+h[s]+h[t])**2/((1+h[r]+h[s])*(1+h[r]+h[t])*(1+h[s]+h[t]))
    
    # sigma_4
    sigma_4 = ((1+4*h)/(1+2*h)**2).sum()
    
    # sigma_5
    sigma_5 = 0
    for r in range(p):
        for s in range(r+1,p):
            sigma_5 = sigma_5 + (1+2*h[r]+2*h[s])/(1+h[r]+h[s])**2
    
    # sigma_6
    sigma_6 = 0
    for r in range(p):
        for s in range(p):
            if s != r:
                sigma_6 = sigma_6 + (1+3*h[r]+h[s])/((1+2*h[r])*(1+h[r]+h[s]))

    # sigma_7
    sigma_7 = 0
    for r in range(p):
        for s in range(p):
            for t in range(p):
                if (s!=r) and (t!=r) and (t!=s):
                    sigma_7 = sigma_7 + (1+2*h[r]+h[s]+h[t])/((1+h[r]+h[s])*(1+h[r]+h[t]))
    
    # rho_23^2
    rho_23_2 = 2*(4*sigma_1 + 3*sigma_2 + 3*sigma_3) / n
    
    # rho_4
    rho_4 = 2*(6*sigma_4 + 3*sigma_5 + 4*sigma_6 + sigma_7)/n
    
    #R_k(T)
    T = rho_4/8 - (3*rho_13_2 + 2*rho_23_2)/24
    return T


@njit
def log_vmf(F):
    """
    Logarithm of the SPA of the normalizing constant for von Mises-Fisher distribution parameterized by F.
    Several code optimizations were added with respect to the original code of Kume et al., based on the following remarks:
    - the matrix V used in Kume et al. is chosen to be the identity in the case of vMF distributions : hence log(det(V))=0 and V@CHat@V = CHat
    - CHat is a diagonal matrix
    """
    
    n, p = F.shape
    lambda_d = np.linalg.svd(F)[1]
    logCorrectionFactor = p/2
    V = np.eye(p*n)
    
    F1 = np.zeros((p, n))
    for i in range(p):
        F1[i,i] = lambda_d[i]
    mu = F1.reshape(-1)
    
    phiHat_d = (-p + np.sqrt(p**2+4*lambda_d**2))/(2*lambda_d**2) # OK
    
    thetaHat_d = 1/2*(1 - 2*lambda_d**2/(np.sqrt(p**2+4*lambda_d**2)-p))
    thetaHat = np.diag(thetaHat_d)
    ThetaTilde_m = 2*thetaHat # OK
    
    CHat_d = np.repeat(1-2*thetaHat_d, n)
    Khat = -.5*np.log(CHat_d).sum() + .5*(mu@(mu/CHat_d)) - .5*(mu@mu) # OK

    K2hat = calc_KHat2_matrixFisher(phiHat_d,lambda_d,n) # OK

    m = 1/2*p*(p+1)
    K2hat_logdet = np.log(np.diag(K2hat)).sum()
    log_f2Hat = -(m/2)*np.log(2*np.pi) - (1/2)*K2hat_logdet + Khat - np.sum(np.diag(thetaHat)) # OK
    log_C2_firstOrder = logCorrectionFactor + log_f2Hat + (n*p/2)*np.log(2*np.pi) + \
        (1/2) * mu@mu + p*np.log(2)
    
    T = compute_T(n, p, lambda_d, phiHat_d)
    res = log_C2_firstOrder + T
    return res
