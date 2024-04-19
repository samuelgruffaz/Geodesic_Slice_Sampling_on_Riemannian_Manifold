# Slice Sampling on Riemannian manifold.

This repository hosts the code for this paper "Slice Sampling on Riemannian manifold".

## Requirements

The code was tested on Python 3.10.9 . In order to run the code, the Python packages listed in `requirements.txt` are needed. They can be installed for instance with `conda`:

```
conda create -n gsv -c conda-forge --file requirements.txt
conda activate gsv
```

On a Apple M1 Pro, 10 heart, 16 Go, the cumulated running time of all experiments does exceed several days.

## Real Data

The real data used in Section 3.2 can be found at this link by creating an account and signing the agreements :
https://www.humanconnectome.org/study/hcp-young-adult/document/extensively-processed-fmri-data-documentation

The related documentation is the following:
https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP1200-DenseConnectome+PT+Appendix-July2017.pdf

The network modeling is related to “netmats” in the documentation.

The KTH video action dataset can be downloaded at this link:
https://www.csc.kth.se/cvap/actions/ 


## Reproducing the experiments

The experiments are separated into different folders, each one containing its own code (user-friendly to push on a cloud). 

The experiments are the following :

- "Geodesics_illustration" contains the codes to see geodesics on the Grassman and the Stiefel manifold in dimension (n,k)=(3,2). 

- "Stiefel_von_Mises_Fisher_distribution" contains the codes for sampling the Mises-Fisher distribution on the Stiefel manifold with GSS and RMH. (Section 3.1)

- "Grassman_von_Mises_Fisher_distribution" contains the codes for sampling the Mises-Fisher distribution on the Grassman manifold with GSS and GeoMALA. (Section 3.1)

- "Networks_analysis_synthetic_estimation" contains the codes for estimating the parameter of the network analysis model by using MCMC-SAEM with GSS and RMH on a synthetic dataset, in order to compare the log complete likelihood and rRMSE to true parameters. (Section 3.2)

- "Networks_analysis_imputation" contains the codes for estimating the parameter of the network analysis model by using MCMC-SAEM with GSS and RMH on a synthetic dataset, in order to recover missing values on masked observations. (3.2)

- "Networks_analysis_real_data" contains the codes for estimating the parameter of the network analysis model by using MCMC-SAEM with GSS and RMH on a real dataset, in order to compare the power of reconstruction. (3.2)

- "ARMA_exp" contains the codes for sampling the posterior related to the ARMA model on the Stiefel manifold with GSS and RMH. (Section 3.3)

- "Clustering_KTH_dataset" contains the codes for clustering video action data using EM algorithm and posterior sampling. (Section 3.4)

Each experiment has its file "exploit_*name_of_the_experiments*.py" to see the results related to the experiments done in the paper, except for Networks_analysis_missing_link_imputation, Networks_analysis_real_data_estimation, and Bayesian_clustering_on_the_KTH_dataset.

 For missing_link_imputation, the results should be printed by running "imputation.py".
  For real_data_estimation, the file is inside the folder, and it is needed to have the real data to run it and to fill the path where the data are saved.
For the clustering, it is the same. You need to download the data and to preprocess it with "Pre_processing.py" in the clustering folder. Once it is done you can directly see our results with "see_results_bayesian.py".

## The codes

The implementation we provide can be used on other densities to sample. The python-related files are "Hybrid_Monte_Carlo.py" for GeoMALA, "Sampler_class.py" and "Sampler_grassman.py" for GSS and RMH on the Stiefel and the Grassman manifold.
The code related to the network analysis is taken from [1] and was changed to add samplers related to GSS.
The code related to clustering is quite specific to the used example but can be readapted for other applications.




[1] Understanding the Variability in Graph Data Sets through Statistical Modeling on the Stiefel Manifold, Clément Mantoux and all, 2021
