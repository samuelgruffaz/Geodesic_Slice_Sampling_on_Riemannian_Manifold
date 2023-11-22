"""
This file contains three categories of functions.

1) SAEM functions
- mcmc_saem implements the main MCMC-SAEM algorithm to estimate the parameters of the base model.
- mcmc_saem_cluster implements the tempered MCMC-SAEM to estimate the parameters of the mixture model.

2) Initialization functions
- init_saem initializes Xs, ls and theta using the eigendecomposition of the adjacency matrices.
- init_saem_grad build on init_saem and perform hybrid MCMC-SAEM steps where the sampling on X is replaced with a gradient ascent step.
- init_saem_cluster executes the K-Means algorithm on the adjacency matrices to initialize the clusters, and call init_saem on each of these clusters.
- init_saem_grad_cluster proceeds as with init_saem_grad for the mixture model.

3) Result permutation functions
- map_to_ground_truth takes the output of the MCMC-SAEM algorithm and permute the columns of F so as to match the ground truth, which allows computing estimation errors.
- map_to_ground_truth_cluster finds the best bijection between the estimated clusters and the true clusters, and align each estimated cluster with its related true cluster as with map_to_ground_truth.


Each of these functions can be used with setting="gaussian" or setting="binary", depending on wether the adjacency matrices have weighted or binary coefficients.


The constants are named as follow:
- n_samples is the number of model samples. In this experiment, n_samples=100.
- n is the number of nodes. In this experiment, n=3.
- p is the number of orthonormal columns. In this experiment, p=2.

The parameter theta is composed of:
- F and mu designate the vMF parameter and the mean eigenvalues
- sigma and sigma_l in the code correspond to sigma_epsilon and sigma_lambda in the paper

The variables used in the code are:
- Xs (n_samples,n,p) is the list of vMF samples (X_1, ..., X_N)
- ls (n_samples,p) is the list of patterns amplitudes for each individual (lambda_1, ..., lambda_N)
- Xs_mh and ls_mh are the MCMC estimate of Xs and ls once the MCMC has converged
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import *
from sklearn.cluster import KMeans

from src.utils import *
import src.stiefel as st
from src import spa, vmf, mcmc, model, model_bin, model_cluster, model_bin_cluster, sampler_slice_rieman


np.random.seed(0)


def mcmc_saem(As, Xs_mh, ls_mh, theta, n_iter=100, mala=False,slice_rieman=False,mh_geo=False, prop_X=0.01, prop_l=0.5, n_mcmc=20, history=True, setting="gaussian",use_adapt=True,m=1):
    #For slice rieman prop_X=w
    F, mu, sigma, sigma_l = theta
    
    optimal_rate = 0.234
    batch = 5 # SAEM steps per column permutation step
    n_mcmc = 20 # MCMC steps per SAEM step
    n_samples, n, p = Xs_mh.shape
    
    # Initialize the exhaustive statistics
    Xs_comp = st.comp(Xs_mh, ls_mh)
    X_bar = Xs_mh.mean(axis=0) 
    l_bar = ls_mh.mean(axis=0)
    l2_bar = (ls_mh**2).mean(axis=0).sum()
    s2_bar = ((As-Xs_comp)**2).mean()

    # Initialize the latent variables
    Xs_mhs = [Xs_mh]
    ls_mhs = [ls_mh]
    Xs_mh = Xs_mh.copy()
    ls_mh = ls_mh.copy()
    lks = []
    # Initialize the parameter history
    Fs = [F]
    mus = [mu]
    sigmas = [sigma]
    sigma_ls = [sigma_l]
    
    As_constants,As_Xs,As_ls=model.IS_proxy(As,p)

    for n in trange(n_iter):
        # MCMC step: use Metropolis-Hastings or MALA
        if mala:
            Xs_mh, ls_mh, lk, rate_X, rate_l = mcmc.mala(As, theta, n_iter=n_mcmc,
                                      init=(Xs_mh, ls_mh), progress=False,
                                      prop_X=prop_X, prop_l=prop_l,
                                      setting=setting)
        elif slice_rieman:
            X_history,Xs_mh, ls_mh, lk, rate_X, rate_l = mcmc.slice_rieman(As, theta, n_iter=n_mcmc,
                                      init=(Xs_mh, ls_mh),
                                      prop_X=prop_X, prop_l=prop_l,
                                      setting=setting,m=m)
        elif mh_geo:
            X_history,Xs_mh, ls_mh, lk, rate_X, rate_l = mcmc.mh_geo(As, theta, n_iter=n_mcmc,
                                      init=(Xs_mh, ls_mh),
                                      prop_X=prop_X, prop_l=prop_l,
                                      setting=setting,m=m)
                                       
        else:
            X_history,Xs_mh, ls_mh, lk, rate_X, rate_l = mcmc.mh(As, theta, n_iter=n_mcmc,
                                      init=(Xs_mh, ls_mh),
                                      prop_X=prop_X, prop_l=prop_l, setting=setting)
        
        if n%batch==0 and n < n_iter/3:
            mode = st.proj_V(F)
            for i in range(n_samples):
                perm, sign = st.greedy_permutation(mode, Xs_mh[i])
                Xs_mh[i] = sign*Xs_mh[i][:,perm]
                ls_mh[i] = ls_mh[i][perm]
        
        # Update proposal variance for adaptive MCMC
        if slice_rieman:
            rate_X=1/rate_X
            if use_adapt:
                adaptive_X = 2*(rate_X > optimal_rate)-1
                prop_X = np.exp(np.log(prop_X) + 0.5*adaptive_X/(1+n)**0.6)
        else:
            adaptive_X = 2*(rate_X > optimal_rate)-1
            prop_X = np.exp(np.log(prop_X) + 0.5*adaptive_X/(1+n)**0.6)
        
        adaptive_l = 2*(rate_l > optimal_rate)-1
        prop_l = np.exp(np.log(prop_l) + 0.5*adaptive_l/(1+n)**0.6)

        # Maximization step

        # Update the stochastic approximation coefficient
        if n < n_iter/2:
            alpha = 1
        else:
            alpha = 1/(n-n_iter/2+1)**0.6

        # Update the exhaustive statistics
        Xs_comp = st.comp(Xs_mh, ls_mh)
        X_bar  = (1-alpha)*X_bar + alpha*Xs_mh.mean(axis=0)
        l_bar  = (1-alpha)*l_bar + alpha*ls_mh.mean(axis=0)
        l2_bar = (1-alpha)*l2_bar + alpha*(ls_mh**2).mean(axis=0).sum()
        s2_bar = (1-alpha)*s2_bar + alpha*((As-Xs_comp)**2).mean()
        
        # Update sigma
        sigma = np.sqrt(s2_bar)
        # Update F
        F = vmf.mle(X_bar, orth=True)
        # Update mu
        mu = l_bar
        # Update sigma_l
        sigma_l = np.sqrt((norm(mu)**2 + l2_bar - 2*(mu*l_bar).sum()) / p)
        
        theta = (F, mu, sigma, sigma_l)
        
        # Store the current complete log-likelihood
        if setting=="gaussian":
            lks.append(model.log_lk(Xs_mh, ls_mh, As, theta, normalized=True))
        elif setting=="binary":
            lks.append(model_bin.log_lk(Xs_mh, ls_mh, As, theta, normalized=True))
        

            
        Fs.append(F)
        mus.append(mu)
        sigmas.append(sigma)
        sigma_ls.append(sigma_l)

        # if history is True, store the values of Xs and ls along the Markov chain
        if history:
            Xs_mhs.append(Xs_mh.copy())
            ls_mhs.append(ls_mh.copy())
            

    print("prop_X",prop_X)
    print("rate",rate_X)
    result = {
        "theta": theta,
        "Xs_mh": Xs_mh,
        "ls_mh": ls_mh,
        "history": {
            "lks": lks,
            "lks_true": model.IS_lk(As,theta,Init=Xs_mh,n_ss=100,stride=10,burn=500),
            "F": Fs,
            "mu": mus,
            "sigma": sigmas,
            "sigma_l": sigma_ls,
            "Xs_mh": Xs_mhs,
            "ls_mh": ls_mhs
        }
    }
    
    return result


def mcmc_saem_cluster(As, Xs_mh, ls_mh, zs_mh, theta, n_iter=100, prop_X=0.01, prop_l=0.5, n_mcmc=20, history=True, setting="gaussian", T=0):
    F, mu, sigma, sigma_l, pi = theta
    
    optimal_rate = 0.234
    batch = 5
    n_samples, n, p = Xs_mh.shape
    K = len(pi)
    
    # Initialize the exhaustive statistics for each cluster
    Xs_comp = st.comp(Xs_mh, ls_mh)
    X_bar = np.zeros((K, n, p))
    l_bar = np.zeros((K, p))
    l2_bar = np.zeros(K)
    s2_bar = np.zeros(K)
    for k in range(K):
        idx = np.where(zs_mh==k)[0]
        # Check that the cluster is not empty
        if len(idx)>0:
            X_bar[k]  = Xs_mh[idx].mean(axis=0)
            l_bar[k]  = ls_mh[idx].mean(axis=0)
            l2_bar[k] = (ls_mh[idx]**2).mean(axis=0).sum()
            s2_bar[k] = ((As[idx]-Xs_comp[idx])**2).mean()
        else:
            X_bar[k]  = Xs_mh.mean(axis=0)
            l_bar[k]  = ls_mh.mean(axis=0)
            l2_bar[k] = (ls_mh**2).mean(axis=0).sum()
            s2_bar[k] = ((As-Xs_comp)**2).mean()
    
    # Initialize the latent variables
    Xs_mh = Xs_mh.copy()
    ls_mh = ls_mh.copy()
    zs_mh = zs_mh.copy().astype(np.int32)
    Xs_mhs = [Xs_mh]
    ls_mhs = [ls_mh]
    zs_mhs = [zs_mh]
    lks = []

    # Initialize the parameter history
    Fs = [F]
    mus = [mu]
    sigmas = [sigma]
    sigma_ls = [sigma_l]
    pis = [pi]
    
    for n in trange(n_iter):
        # MCMC step
        temp = 1+T/(n+1)**0.6
        Xs_mh, ls_mh, zs_mh, _, rate_X, rate_l = mcmc.mh_cluster(As, theta, n_iter=n_mcmc,
                                      init=(Xs_mh, ls_mh, zs_mh),
                                      prop_X=prop_X, prop_l=prop_l, setting=setting, T=temp)
        
        if n%batch==0:
            mode = [st.proj_V(F[k]) for k in range(K)]
            mu_old = mu.copy()
            # Align the F parameters of each cluster to the first cluster.
            for k in range(1,K):
                perm, sign = st.greedy_permutation(mode[0], mode[k])
                F[k] = sign*F[k][:,perm]
                mu[k] = mu[k][perm]
            # Permute the X columns to best match the F parameter of their cluster
            if n < n_iter//3 or norm(mu_old-mu)>0:
                for i in range(n_samples):
                    perm, sign = st.greedy_permutation(mode[zs_mh[i]], Xs_mh[i])
                    Xs_mh[i] = sign*Xs_mh[i][:,perm]
                    ls_mh[i] = ls_mh[i][perm]

        
        # Update proposal variance for adaptive MCMC
        adaptive_X = 2*(rate_X > optimal_rate)-1
        prop_X = np.exp(np.log(prop_X) + 0.5*adaptive_X/(1+n)**0.6)
        adaptive_l = 2*(rate_l > optimal_rate)-1
        prop_l = np.exp(np.log(prop_l) + 0.5*adaptive_l/(1+n)**0.6)

        # Maximization step

        # Update the stochastic approximation coefficient
        if n < n_iter/2:
            alpha = 1
        else:
            alpha = 1/(n-n_iter/2+1)**0.6

        # Update the exhaustive statistics
        Xs_comp = st.comp(Xs_mh, ls_mh)
        for k in range(K):
            idx = np.where(zs_mh==k)[0]
            if len(idx)>0:
                X_bar_new  = Xs_mh[idx].mean(axis=0)
                l_bar_new  = ls_mh[idx].mean(axis=0)
                l2_bar_new = (ls_mh[idx]**2).mean(axis=0).sum()
                s2_bar_new = ((As[idx]-Xs_comp[idx])**2).mean()
                
                X_bar[k]  = (1-alpha)*X_bar[k] + alpha*X_bar_new
                l_bar[k]  = (1-alpha)*l_bar[k] + alpha*l_bar_new
                l2_bar[k] = (1-alpha)*l2_bar[k] + alpha*l2_bar_new
                s2_bar[k] = (1-alpha)*s2_bar[k] + alpha*s2_bar_new
        
        # Update the parameters for each cluster
        for k in range(K):
            sigma[k] = np.sqrt(s2_bar[k])
            F[k] = vmf.mle(X_bar[k], orth=True)
            mu[k] = l_bar[k]
            sigma_l[k] = np.sqrt((norm(mu[k])**2+l2_bar[k]-2*(mu[k]*l_bar[k]).sum()) / p)
        pi = pi.copy()
        for k in range(K):
            pi[k] = (zs_mh==k).mean()
        
        theta = (F, mu, sigma, sigma_l, pi)
        
        # Store the current complete log-likelihood
        if setting=="gaussian":
            lks.append(model_cluster.log_lk(Xs_mh, ls_mh, zs_mh, As, theta, normalized=True))
        elif setting=="binary":
            lks.append(model_bin_cluster.log_lk(Xs_mh, ls_mh, zs_mh, As, theta, normalized=True))

        Fs.append(F)
        mus.append(mu)
        sigmas.append(sigma)
        sigma_ls.append(sigma_l)
        pis.append(pi)
        
        # if history is True, store the values of Xs and ls along the Markov chain
        if history:
            Xs_mhs.append(Xs_mh.copy())
            ls_mhs.append(ls_mh.copy())
            zs_mhs.append(zs_mh.copy())

    result = {
        "theta": theta,
        "Xs_mh": Xs_mh,
        "ls_mh": ls_mh,
        "zs_mh": zs_mh,
        "history": {
            "lks": lks,
            "F": Fs,
            "mu": mus,
            "sigma": sigmas,
            "sigma_l": sigma_ls,
            "pi": pis,
            "Xs_mh": Xs_mhs,
            "ls_mh": ls_mhs,
            "zs_mh": zs_mhs
        }
    }
    
    return result


def init_saem(As, p):
    n_samples, n, _ = As.shape
    ls = np.zeros((n_samples, p))
    Xs = np.zeros((n_samples, n, p))
    # Compute the eigendecomposition of each adjacency matrix
    for i in range(n_samples):
        ev, u = np.linalg.eig(As[i])
        idx = (-np.abs(ev)).argsort()[:p]
        ls[i] = ev[idx]
        Xs[i] = u[:,idx]
    # Average on the eigenvectors on the Stiefel manifold
    mode = st.proj_V(Xs.mean(axis=0))
    # Permute the eigenvectors to align them with the computed mode
    for i in range(n_samples):
        m, s = st.greedy_permutation(mode, Xs[i])
        Xs[i] = s*Xs[i][:,m]
        ls[i] = ls[i][m]
    # Initialize the parameters from the resulting eigenvectors and eigenvalues
    F = vmf.mle(Xs.mean(axis=0))
    mu = ls.mean(axis=0)
    sigma = (As-st.comp_numba_many(Xs, ls)).std()
    sigma_l = (ls-mu).std()
    return (F, mu, sigma, sigma_l), Xs, ls


def init_saem_grad(As, p, n_iter=10, step=0.1, setting="gaussian"):
    global model
    if setting=="binary":
        model = model_bin
    n_samples, n, _ = As.shape
    theta, _, _ = init_saem(As, p)
    F, mu, sigma, sigma_l = theta
    sigma = 1
    sigma_l = 1
    mode = st.proj_V(F)
    Xs = np.array([mode.copy() for _ in range(n_samples)])
    ls = mu[None,:] + sigma_l*np.random.randn(n_samples, p)
    lks = []
    it = trange(n_iter)
    prop_l = 1
    current_log_lk = np.array([model.log_lk_partial(Xs[i], ls[i], As[i], theta) for i in range(n_samples)])
    for t in it:
        mode = st.proj_V(F)
        posterior_std_l = 1/(1/sigma**2 + 1/sigma_l**2)
        for _ in range(10):
            for i in range(n_samples):
                if t%5==0:
                    m, s = st.greedy_permutation(mode, Xs[i])
                    Xs[i] = s*Xs[i][:,m]
                    ls[i] = ls[i][m]

                grad_X = model.log_lk_partial_grad_X(Xs[i], ls[i], As[i], theta)
                grad_X = grad_X/norm(grad_X)
                Xs[i] = st.proj_V(Xs[i]+step*grad_X)

                # [l] Generate next move
                l2 = ls[i] + prop_l*np.random.randn(p)
                # [l] Compute the acceptance log-probability
                new_log_lk = model.log_lk_partial(Xs[i], l2, As[i], theta)
                log_alpha = new_log_lk - current_log_lk[i]
                # [l] Accept or reject
                if np.log(np.random.rand()) < log_alpha:
                    ls[i] = l2
                    current_log_lk[i] = new_log_lk
        
        F = vmf.mle(Xs.mean(axis=0))
        mu = ls.mean(axis=0)
        sigma = ((As-st.comp_numba_many(Xs,ls))**2).mean()
        sigma_l = ((ls-mu)**2).mean()
        theta = (F, mu, sigma, sigma_l)
        lks.append(model.log_lk(Xs, ls, As, theta, normalized=True))
        it.set_postfix({"lk": lks[-1]})
    return theta, Xs, ls, lks


def init_saem_cluster(As, p, K, n_iter=100, step=0.1, setting="gaussian"):
    n_samples, n, _ = As.shape
    kmeans = KMeans(n_clusters=K).fit(As.reshape(n_samples, -1))
    zs = kmeans.labels_
    
    F = np.zeros((K, n, p))
    mu = np.zeros((K, p))
    sigma = np.zeros(K)
    sigma_l = np.zeros(K)
    pi = np.bincount(zs)/n_samples
    Xs = np.zeros((n_samples, n, p))
    ls = np.zeros((n_samples, p))
    for k in range(K):
        idx = np.where(zs==k)[0]
        (F[k], mu[k], sigma[k], sigma_l[k]), Xs[idx], ls[idx] = init_saem(As[idx], p)
    
    return (F, mu, sigma, sigma_l, pi), Xs, ls, zs


def init_saem_grad_cluster(As, p, K, n_iter=10, step=0.1, setting="gaussian"):
    n_samples, n, _ = As.shape
    kmeans = KMeans(n_clusters=K).fit(As.reshape(n_samples, -1))
    zs = kmeans.labels_
    
    F = np.zeros((K, n, p))
    mu = np.zeros((K, p))
    sigma = np.zeros(K)
    sigma_l = np.zeros(K)
    pi = np.bincount(zs)/n_samples
    for k in range(K):
        idx = np.where(zs==k)[0]
        (F[k], mu[k], sigma[k], sigma_l[k]), _, _ = init_saem(As[idx], p)
    
    mode = [st.proj_V(F[k]) for k in range(K)]
    Xs = np.array([mode[zs[i]].copy() for i in range(n_samples)])
    ls = mu[zs]
    
    lks = []
    prop_l = 1
    it = trange(n_iter)
    current_log_lk = np.array([model.log_lk_partial(Xs[i], ls[i], As[i], (F[zs[k]], mu[zs[k]], sigma[zs[k]], sigma_l[zs[k]])) for i in range(n_samples)])
    for t in it:
        mode = [st.proj_V(F[k]) for k in range(K)]
        posterior_std_l = 1/(1/sigma**2 + 1/sigma_l**2)
        for _ in range(10):
            for i in range(n_samples):
                if t%5==0:
                    m, s = st.greedy_permutation(mode[k], Xs[i])
                    Xs[i] = s*Xs[i][:,m]
                    ls[i] = ls[i][m]
                
                k = zs[i]
                theta = (F[k], mu[k], sigma[k], sigma_l[k])
                
                if setting=="gaussian":
                    grad_X = model.log_lk_partial_grad_X(Xs[i], ls[i], As[i], theta)
                elif setting=="binary":
                    grad_X = model_bin.log_lk_partial_grad_X(Xs[i], ls[i], As[i], theta)
                grad_X = grad_X/norm(grad_X)
                Xs[i] = st.proj_V(Xs[i]+step*grad_X)

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

        for k in range(K):
            idx = np.where(zs==k)[0]
            F[k] = vmf.mle(Xs[idx].mean(axis=0))
            mu[k] = ls[idx].mean(axis=0)
            sigma[k] = ((As[idx]-st.comp_numba_many(Xs[idx],ls[idx]))**2).mean()
            sigma_l[k] = ((ls[idx]-mu[k])**2).mean()
        
        if setting=="gaussian":
            lks.append(model_cluster.log_lk(Xs, ls, zs, As, (F, mu, sigma, sigma_l, pi), normalized=True))
        elif setting=="binary":
            lks.append(model_cluster.log_lk(Xs, ls, zs, As, (F, mu, sigma, sigma_l, pi), normalized=True))
        
        it.set_postfix({"lk": lks[-1]})
    return (F, mu, sigma, sigma_l, pi), Xs, ls, zs, lks


def map_to_ground_truth(Xs_mh, ls_mh, theta, theta0):
    F, mu, sigma, sigma_l = theta
    F0, mu0, sigma0, sigma_l0 = theta0
    m, s = st.greedy_permutation(st.proj_V(F0), st.proj_V(F))
    F = s*F[:,m]
    mu = mu[m]
    Xs_mh = s*Xs_mh[:,:,m]
    ls_mh = ls_mh[:,m]
    return Xs_mh, ls_mh, (F, mu, sigma, sigma_l), m, s


def map_to_ground_truth_cluster(Xs_mh, ls_mh, zs_mh, theta, theta0, M=None):
    F, mu, sigma, sigma_l, pi = theta
    F0, mu0, sigma0, sigma_l0, pi0 = theta0
    n_samples, n, p = Xs_mh.shape
    K = len(pi)
    ms = np.zeros((K, K, p), dtype=np.int32)
    ss = np.zeros((K, K, p))
    E = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            X = st.proj_V(F0[k])
            Y = st.proj_V(F[l])
            ms[k,l], ss[k,l] = st.greedy_permutation(X, Y)
            E[k,l] += st.discr(F0[k], ss[k,l]*F[l][:,ms[k,l]])
            E[k,l] += np.linalg.norm(mu0[k]-mu[l][ms[k,l]])
    
    if M is None:
        M = np.zeros(K, dtype=np.int32)
        iM = np.zeros(K, dtype=np.int32)
        for k in range(K):
            k, l = np.unravel_index(E.argmin(), E.shape)
            E[k, :] = 0
            E[:, l] = 0
            M[k] = l
            iM[l] = k
    else:
        iM = np.zeros(K, dtype=np.int32)
        for k, l in enumerate(M):
            iM[l] = k
    
    pi = pi[M]
    ms = [ms[k,M[k]] for k in range(K)]
    ss = [ss[k,M[k]] for k in range(K)]

    F_ = F.copy()
    mu_ = mu.copy()
    for k in range(K):
        m = ms[k]
        s = ss[k]
        F[k] = s*F_[M[k]][:,m]
        mu[k] = mu_[M[k]][m]
    
    zs_mh = np.array([iM[zs_mh[i]] for i in range(n_samples)])
    for i in range(n_samples):
        m = ms[zs_mh[i]]
        s = ss[zs_mh[i]]
        Xs_mh[i] = s*Xs_mh[i][:,m]
        ls_mh[i] = ls_mh[i][m]
    return Xs_mh, ls_mh, zs_mh, (F, mu, sigma, sigma_l, pi), ms, ss
