import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

svd = np.linalg.svd
norm = np.linalg.norm

def sv(A):
    """Singular values of matrix A"""
    return np.linalg.svd(A, compute_uv=False)


def no_axis():
    """Remove the axes from a matplotlib image plot"""
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')


def hard_threshold(X, tau):
    """Truncates the coefficients of X with absolute value lower than tau"""
    a = X.copy()
    a[np.abs(a) < tau] = 0
    return a


def hard_nuclear(A, tau):
    """Truncates the singular values of A under absolute threshold tau"""
    u, s, v = svd(A)
    return u@np.diag(hard_threshold(s, tau))@v


def relative_error(A, B, order=None, rnd=300):
    """Relative Root Mean Square Error"""
    return (norm(A-B, order)/norm(B, order)).round(rnd)