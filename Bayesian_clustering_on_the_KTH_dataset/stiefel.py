"""
This script provides miscellaneous functions for handling matrices in the Stiefel manifold V(n,p).
"""


import numpy as np
import scipy.linalg
from numba import njit


def dim(n, p):
    """Dimension of V(n,p)"""
    return n*p - p*(p+1)//2


def unif(n, p):
    """Uniform sampling on the V(n,p)"""
    O = scipy.stats.ortho_group.rvs(n)
    return O[:,:p]


def unif_like(X):
    n, p = X.shape
    return unif(n, p)


def get_X_(X):
    """Find a (n,n-p) matrix X_ such that (X, X_) is an orthogonal matrix"""
    p = X.shape[1]
    Q, R = scipy.linalg.qr(X, mode="full")
    X_ = Q[:,p:]
    return X_


def getAB(X, D):
    X_ = get_X_(X)
    return X.T@D, X_.T@D


def discr(X, Y):
    """Squared Euclidean distance between X and Y."""
    n, p = X.shape
    return 2*(p-(X*Y).sum())


def comp(X, l=None):
    """
    If X is a (n,p) matrix and l a (p) vector, returns X@np.diag(l)@X.T.
    It also works element-wise if X and l are arrays of such matrices and vectors.
    """
    if l is None: l = np.ones(X.shape[-1])
    return np.einsum("...ij,...j,...kj->...ik", X, l, X)
@njit
def Ax_numba(A,X):
    """
    Numba version of comp, for inputs with shape [n,d], [d]
    """
    n, d = X.shape
    result = np.zeros(d)
    for i in range(d):
        result[i]=(A.dot(X[:,i])).dot(X[:,i])
    return result

@njit
def comp_numba_single(X, l=None):
    """
    Numba version of comp, for inputs with shape [n,d], [d]
    """
    n, d = X.shape
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for m in range(d):
                result[i,j] += l[m] * X[i,m] * X[j,m]
    return result


@njit
def comp_numba_many(X, l=None):
    """
    Numba version of comp, for inputs with shape [N,n,d], [N,d]
    """
    N, n, d = X.shape
    result = np.zeros((N, n, n))
    for t in range(N):
        for i in range(n):
            for j in range(n):
                for m in range(d):
                    result[t,i,j] += l[t,m] * X[t,i,m] * X[t,j,m]
    return result


def skew(A):
    """Return the skew-symmetric part of a square matrix A"""
    return (A-A.T)/2


@njit
def proj_V(X):
    """Orthogonal projection onto V(n,p) for a matrix X with full rank"""
    u, _, v = np.linalg.svd(X, full_matrices=False)
    return u@v


def proj_T(X, D):
    """Project D onto the tangent space of V(n,p) at point X"""
    return X@skew(X.T@D) + (np.eye(X.shape[0])-X@X.T)@D


def scal_T(X, D1, D2):
    """Riemannian scalar product of tangent vectors D1 and D2 at point X"""
    A1, B1 = getAB(X, D1)
    A2, B2 = getAB(X, D2)
    return (A1*A2).sum()/2 + (B1*B2).sum()


def norm_T(X, D):
    """Riemannian norm of tangent vectors D at point X"""
    return np.sqrt(scal_T(X, D, D))


def greedy_permutation(X, Y):
    """
    For X, Y elements of V(n,p), finds a permutation and sign arrangement
    of the columns of Y that best matches the columns of X.
    """
    n, p = X.shape
    D = X.T@Y
    E = np.abs(D)
    mapping = np.zeros(p, dtype=np.int)
    sign = np.zeros(p)
    for i in range(p):
        g, h = np.unravel_index(E.argmax(), E.shape)
        E[g, :] = 0
        E[:, h] = 0
        mapping[g] = h
        sign[g] = np.sign(D[g,h])
    return mapping, sign
