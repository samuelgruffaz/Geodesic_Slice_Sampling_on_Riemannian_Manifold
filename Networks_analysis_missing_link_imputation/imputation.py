"""
This script reproduces the experiment of section 3.2 on missing link imputation.
The constants are named as follow:
- n_samples is the number of model samples. In this experiment, n_samples=200.
- n is the number of nodes. In this experiment, n=20.
- p is the number of orthonormal columns. In this experiment, p=5.

The parameter theta is composed of:
- F and mu designate the vMF parameter and the mean eigenvalues
- sigma and sigma_l in the code correspond to sigma_epsilon and sigma_lambda in the paper

The variables used in the code are:
- Xs (n_samples,n,p) is the list of vMF samples (X_1, ..., X_N)
- ls (n_samples,p) is the list of patterns amplitudes for each individual (lambda_1, ..., lambda_N)
- Xs_mh and ls_mh are the MCMC estimate of Xs and ls once the MCMC has converged
"""


import numpy as np
from matplotlib.pyplot import *
from tqdm.auto import *
import os

from src.utils import *
from src.stiefel import *
from src import spa, vmf, mcmc, model, saem

np.random.seed(0)
set_cmap("bwr")
fig_folder = "figures"
try:
    os.mkdir(fig_folder)
except:
    pass


#==============================
# TRAIN ON A FIRST DATA SET
#==============================

print("Experiment: missing link imputation.")
print("Generating the synthetic training data set.")

F0 = [60,20,20,20,5]*unif(20,5)
mu0 = np.array([20,10,5,2,-10], dtype=np.float)

sigma0 = 0.3 # sigma_epsilon
sigma_l0 = 2 # sigma_lambda
n_samples = 200

u, s, v = svd(F0, full_matrices=False)
u *= np.sign(u[0])
F0 = u@np.diag(s)@v
mu0 = mu0[np.abs(mu0).argsort()][::-1]
theta0 = (F0, mu0, sigma0, sigma_l0)
mode0 = proj_V(F0)
n_nodes = F0.shape[0]
n, p = F0.shape

ls = mu0[None,:].repeat(n_samples, axis=0)
ls += sigma_l0*np.random.randn(*ls.shape)
Xs = vmf.sample_von_mises_fisher(F0, n_iter=n_samples, burn=10000, stride=10)
As = comp(Xs, ls)
idx = np.triu_indices(F0.shape[0])
noise = sigma0*np.random.randn(*As.shape)
noise[:,idx[0],idx[1]] = noise[:,idx[1],idx[0]]
As += noise



print("Initializing the MCMC-SAEM algorithm.")
theta, Xs_mh, ls_mh, lks = saem.init_saem_grad(As, p=5, n_iter=20, step=0.05)

# Run the MCMC-SAEM for 2000 iterations with 20 MCMC steps per SAEM step
print("Running the MCMC-SAEM algorithm.")
result = saem.mcmc_saem(As, Xs_mh, ls_mh, theta, n_mcmc=20, n_iter=2000,prop_X=1,slice_rieman=True,use_adapt=False,m=10,likelihood=False)

# Open results
theta = result["theta"]
F, mu, sigma, sigma_l = theta
m, s = greedy_permutation(proj_V(F0), proj_V(F))
F = s*F[:,m]
mu = mu[m]
Xs_mh = s*result["Xs_mh"][:,:,m]
ls_mh = result["ls_mh"][:,m]
mode = proj_V(F)

print("\n\n")

#========================================
# IMPUTATION OF A BLOCK OF COEFFICIENTS
#========================================

print("Generating the synthetic test data set.")
np.random.seed(1) # Use a different random seed than the training set
ls2 = mu0[None,:].repeat(n_samples, axis=0)
ls2 += sigma_l0*np.random.randn(*ls2.shape)
Xs2 = vmf.sample_von_mises_fisher(F0, n_iter=n_samples, burn=10000, stride=10)
As2 = comp(Xs2, ls2)
idx = np.triu_indices(F0.shape[0])
noise = sigma0*np.random.randn(*As2.shape)
noise[:,idx[0],idx[1]] = noise[:,idx[1],idx[0]]
As2 += noise


error_map = []
error_pmean = []
error_mean = []
error_comp = []

# Build mask
h = 8 # Number of nodes involved in the mask
I = np.arange(n)[:,None].repeat(n,1)
J = I.T
mask = np.where((I>=n-h)*(J>=n-h)) # Hide coefficients in the lower right corner

print("Performing missing link imputation of a block of coefficients over test samples.")
for i in trange(n_samples):
    A_init = As2[i]
    A = A_init.copy()
    A[mask[0], mask[1]] = 0
    A[mask[1], mask[0]] = 0

    # Compute MAP
    n_iter = 4000
    A_map, X_map, l_map, lks_map = mcmc.map_mask(A, mask, theta, n_iter=n_iter)

    # Compute posterior mean
    n_iter = 10000
    #we change here the sampler compared to the orginal paper, slice_mask instead of mh_mask
    A_mcmc, X_mcmc, l_mcmc, lks_mcmc = mcmc.slice_mask(A, mask, theta, n_iter=n_iter,prop_X=1,m=10)
    A_pmean = A_mcmc[n_iter//2:].mean(axis=0)
    X_pmean = proj_V(X_mcmc[n_iter//2:].mean(axis=0))
    l_pmean = l_mcmc[n_iter//2:].mean(axis=0)
    
    # Compute rRMSE, on the missing links
    error_map.append(relative_error(A_map[mask], As2[i][mask], rnd=3))
    error_pmean.append(relative_error(A_pmean[mask], As2[i][mask], rnd=3))
    error_mean.append(relative_error(As2.mean(axis=0)[mask], As2[i][mask], rnd=3))
    error_comp.append(relative_error(comp(Xs2[i], ls2[i])[mask], As2[i][mask], rnd=3))


print("Relative Root Mean Square Error:")
print("Posterior mean      :", np.mean(error_pmean).round(2), f"(+/- {np.std(error_pmean).round(2)})")
print("MAP                 :", np.mean(error_map).round(2), f"(+/- {np.std(error_map).round(2)})")
print("Mean of samples     :", np.mean(error_mean).round(2), f"(+/- {np.std(error_mean).round(2)})")
print("Sample without noise:", np.mean(error_comp).round(2), f"(+/- {np.std(error_comp).round(2)})")


#==============================
# FIGURE ON A SPECIFIC SAMPLE
#==============================

np.random.seed(0)
i = np.random.randint(n_samples)
# i = 23
mean = As2.mean(axis=0)
A_init = As2[i].copy()
A = A_init.copy()
A[mask[0], mask[1]] = 0
A[mask[1], mask[0]] = 0

# Compute MAP
n_iter = 4000
A_map, X_map, l_map, lks_map = mcmc.map_mask(A, mask, theta, n_iter=n_iter)

# Compute posterior mean
n_iter = 100000
#we change the sampler again
A_mcmc, X_mcmc, l_mcmc, lks_mcmc = mcmc.slice_mask(A, mask, theta, n_iter=n_iter,prop_X=1,m=10)
A_pmean = A_mcmc[n_iter//2:].mean(axis=0)
X_pmean = proj_V(X_mcmc[n_iter//2:].mean(axis=0))
l_pmean = l_mcmc[n_iter//2:].mean(axis=0)

font = 16
M = 6
def add_line(h):
    dx = 0.1
    plot([n-h-.5,n-h-.5,n-0.5-dx,n-0.5-dx,n-h-.5],[n-0.7,n-h-.5,n-h-.5,n-0.5-dx,n-0.5-dx], c="black", alpha=1, linewidth=2)

figure(figsize=(20,4))
subplots_adjust(wspace=0)

subplot(141)
title(f"(a)\nrRMSE={relative_error(A_init[mask], A_init[mask], rnd=2)}", fontsize=font)
imshow(A_init, vmin=-M, vmax=M); no_axis()
add_line(h)
colorbar()

subplot(142)
title(f"(b)\nrRMSE={relative_error(A_pmean[mask], A_init[mask], rnd=2)}", fontsize=font)
imshow(A_pmean, vmin=-M, vmax=M); no_axis()
add_line(h)
colorbar()

subplot(143)
title(f"(c)\nrRMSE={relative_error(A_map[mask], A_init[mask], rnd=2)}", fontsize=font)
imshow(A_map, vmin=-M, vmax=M); no_axis()
add_line(h)
colorbar()

subplot(144)
title(f"(d)\nrRMSE={relative_error(mean[mask], A_init[mask], rnd=2)}", fontsize=font)
imshow(mean, vmin=-M, vmax=M); no_axis()
add_line(h)
colorbar()

fig_path = fig_folder+"/imputation.pdf"
savefig(fig_path)
print(f"Saved figure at {fig_path}.")


print("\n\n")
#=============================================
# IMPUTATION OF RANDOMLY MASKED COEFFICIENTS
#=============================================

np.random.seed(0)

error2_map = []
error2_pmean = []
error2_mean = []
error2_comp = []

# Build mask with 160 randomly selected edges
idx = np.triu_indices(n)
coefs_idx = np.random.choice(np.arange(len(idx[0])), size=100, replace=False)
mask = [[],[]]
k = 0
while len(mask[0]) < 160:
    i, j = idx[0][coefs_idx[k]], idx[1][coefs_idx[k]]
    mask[0].append(i)
    mask[1].append(j)
    if i!=j:
        mask[0].append(j)
        mask[1].append(i)
    k += 1
mask = np.array(mask)

print("Performing missing link imputation of a block of coefficients over test samples.")

for i in trange(n_samples):
    A_init = As2[i]
    A = A_init.copy()
    A[mask[0], mask[1]] = 0
    A[mask[1], mask[0]] = 0

    # Compute MAP
    n_iter = 4000
    A_map, X_map, l_map, lks_map = mcmc.map_mask(A, mask, theta, n_iter=n_iter)

    # Compute posterior mean
    n_iter = 10000
    A_mcmc, X_mcmc, l_mcmc, lks_mcmc = mcmc.slice_mask(A, mask, theta, n_iter=n_iter,prop_X=1,m=10)
    A_pmean = A_mcmc[n_iter//2:].mean(axis=0)
    X_pmean = proj_V(X_mcmc[n_iter//2:].mean(axis=0))
    l_pmean = l_mcmc[n_iter//2:].mean(axis=0)
    
    # Compute rRMSE
    error2_map.append(relative_error(A_map[mask], As2[i][mask], rnd=3))
    error2_pmean.append(relative_error(A_pmean[mask], As2[i][mask], rnd=3))
    error2_mean.append(relative_error(As2.mean(axis=0)[mask], As2[i][mask], rnd=3))
    error2_comp.append(relative_error(comp(Xs2[i], ls2[i])[mask], As2[i][mask], rnd=3))


print("Relative Root Mean Square Error:")
print("Posterior mean      :", np.mean(error2_pmean).round(2), f"({np.std(error2_pmean).round(2)})")
print("MAP                 :", np.mean(error2_map).round(2), f"({np.std(error2_map).round(2)})")
print("Mean of samples     :", np.mean(error2_mean).round(2), f"({np.std(error2_mean).round(2)})")
print("Sample without noise:", np.mean(error2_comp).round(2), f"({np.std(error2_comp).round(2)})")