import numpy as np
import spa, vmf
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from Sampler_class import Sampler_Slice_Sampling_Geodesics
from sklearn.utils.extmath import fast_logdet
from sklearn.cluster import KMeans


#theta = (V_npxS_pxV_pp)^C in the ref paper
# for computation we prefer theta= (F)^C





def predict_label_with_Z(Z):
    Labels=np.argmax(Z,axis=1)
    return Labels
        
def MCMC_posterior_F_clustering(X,F,Z,n_mcmc=1000,n_burn=100):# pas sample sur pi
    """
    Return the samples of the pseudo-posterior sampling of p(F|X,Z,pi) (pseudo since we are considering the posterior on some part of the svd of F)
    where F is of size (n_cluster,n,p) with n,p the dimension of the Stiefel.
    """
    n_cluster=len(F)
    n,p=F[0].shape
    n_samples=(n_mcmc-n_burn)//2 # tocheck
    F_samples=np.zeros((n_cluster,n_samples,n,p))
    for i in range(n_cluster):# mean posterior on the F_i of each cluster
        X_pond=np.exp(Z[:,i])[:,None,None]*X
        X_pond_sum=X_pond.sum(axis=0)
        F_samp=MCMC_F_cluster(F[i],X_pond_sum,n_mcmc=n_mcmc,n_burn=n_burn)
        if i==0:
            F_samples=np.zeros((n_cluster,len(F_samp),n,p))
        F_samples[i]=F_samp

    
    return F_samples


def MCMC_F_cluster(F_i,X_pond_sum,n_mcmc=1000,n_burn=100):
    """
    Perform MCMC steps on the posterior related to U_i where F_i=U_iD_V_i^T is the svd.
    X_pond_sum enables to parametrize the posterior which is a Mises-Fisher distribution
    """
    U, D, V = np.linalg.svd(F_i, full_matrices=False)
    U_post_param=np.diag(D)@V@X_pond_sum.T# The posterior is a mises fisher distribution with this parameter
    log_prob=lambda x: (U_post_param.T*x).sum()
    n,p=U.shape
    sampler=Sampler_Slice_Sampling_Geodesics(log_prob,n,p)
    samples=sampler.run_kernel(X_0=U.copy(), n_iter=n_mcmc,w=3,m=3)
    samples_with_burn=samples[n_burn:]
    Id=[i for i in range(0,len(samples_with_burn),2)]
    vec=np.zeros((len(Id),n,p))
    for k,i in enumerate(Id):
        vec[k]=(samples_with_burn[i]*D)@V
    
    return vec

def Bayesian_weights(X1,X2,X3,log_pi,F1,mu,cov,F3,Z,n_mcmc=1000,n_burn=100):
    """
    First sample the pseudo posterior on the Mises Fisher distribution parameter,
    and then use these samples to evaluate the cluster assigments weights related to each 
    observations (X1,X2,X3) (weights), then compute their mean (weights_mean) and their std (weights_std)
    """
    F1_samples=MCMC_posterior_F_clustering(X1,F1,Z,n_mcmc=n_mcmc,n_burn=n_burn)
    F3_samples=MCMC_posterior_F_clustering(X3,F3,Z,n_mcmc=n_mcmc,n_burn=n_burn)
    nb_cluster,N_samples,n,p=F1_samples.shape
    N=len(X1)
    weights=np.zeros((N_samples,N,nb_cluster))
    for i in range(N_samples):
        weights[i],log_=compute_Z_pivot(X1,X2,X3,log_pi,F1_samples[:,i,:,:],mu,cov,F3_samples[:,i,:,:])
    weights_mean=weights.mean(axis=0)
    weights_std=weights.std(axis=0)

    return weights_mean,weights_std,weights

def Bayesian_weights_test(X1,X2,X3,X1_test,X2_test,X3_test,log_pi,F1,mu,cov,F3,Z,n_mcmc=1000,n_burn=100):
    """
    First sample the pseudo posterior on the Mises Fisher distribution parameter,
    and then use these samples to evaluate the cluster assigments weights related to each 
    observations on the test set (X1_test,X2_test,X3_test) (weights), then compute their mean (weights_mean) and their std (weights_std)
    """
    F1_samples=MCMC_posterior_F_clustering(X1,F1,Z,n_mcmc=n_mcmc,n_burn=n_burn)
    F3_samples=MCMC_posterior_F_clustering(X3,F3,Z,n_mcmc=n_mcmc,n_burn=n_burn)
    nb_cluster,N_samples,n,p=F1_samples.shape
    N=len(X1)
    weights=np.zeros((N_samples,len(X1_test),nb_cluster))
    for i in range(N_samples):
        weights[i],log_=compute_Z_pivot(X1_test,X2_test,X3_test,log_pi,F1_samples[:,i,:,:],mu,cov,F3_samples[:,i,:,:])
    weights_mean=weights.mean(axis=0)
    weights_std=weights.std(axis=0)

    return weights_mean,weights_std,weights

    


def log_gaussian(X,mu,cov,normalized=False):
    """
    Log gaussian density of parameter mu,cov on the vector X
    """
    d=len(mu)
    part1=-(X-mu).dot(np.linalg.inv(cov+10**(-4)*np.eye(d))@(X-mu))/2
    if normalized:
        return part1-fast_logdet(cov)/2-d*np.log(2*np.pi)/2
    else:
        return part1

def prod_EM_estimation(X1,X2,X3,Label,nb_cluster,n_iter=100,kmens=True,diag=False,init=None):# X data
    """
    (X_1,X_2,X_3) are the observations, 
    Label their label to compute the score during the estimation,
    nb_cluster the number of cluster (in the experiments =6)
    n_iter the number of (EM) steps,
    kmeans =True to use kmeans as initialization for the labels
    diag=True to use Gaussian mixture with diagonal covariance matrix
    init= to give a specific initialization for the label

    Return the parameters log_pi,F1,mu,cov,F3 of the model by performing the EM algorithm
    Z-> weights of clusser assignements,
    Error,Score_list-> list of indicators during the optimization
    
    """
    N_obs=len(X1)
    n,p=X1[0].shape #X1[0] of size nxp
    #X3[0] of size pxp and X_2 (N_obs,p)
   
    # initialisation of latent Z with k means or gaussian mixture with diagonal covariacne matrix
    if init is None:
        X1_resize=np.reshape(X1,(N_obs,-1))
        X3_resize=np.reshape(X3,(N_obs,-1))
        X_full=np.concatenate([X1_resize,X2,X3_resize],axis=1)
        print("begin init")
        if kmens:
            gm = KMeans(n_clusters=nb_cluster).fit(X_full)# à changer avec kmeans si c'est trop long
        else:
            gm= GaussianMixture(n_components=nb_cluster,covariance_type='diag').fit(X_full)
        print("fit init ok")
        Label_predict=gm.predict(X_full)
        
        del X_full
        del X3_resize
        del X1_resize
    else:#Use the predefined labels
        Label_predict=init.copy()
        
    #Define the assignments weights
    Z_0=np.zeros((N_obs,nb_cluster))
    for i in range(N_obs):
        Z_0[i,int(Label_predict[i])]=1
    log_pi=np.log(Z_0.sum(axis=0)/N_obs)

    F1=np.zeros((nb_cluster,n,p))
    mu=np.zeros((nb_cluster,p))
    cov=np.zeros((nb_cluster,p,p))
    F3=np.zeros((nb_cluster,p,p))
    #compute the parameter related to each cluster
    print("begin estimation init")
    for i in range(nb_cluster):
        average1=X1[Z_0[:,i].astype(np.bool)].sum(axis=0)/Z_0[:,i].sum()
        F1[i]=vmf.mle(average1)#fit of mises fisher on each cluster
        F3[i]=vmf.mle(X3[Z_0[:,i].astype(np.bool)].sum(axis=0)/Z_0[:,i].sum())
        mu[i]=np.mean(X2[Z_0[:,i].astype(np.bool)],axis=0)
        Piv=mu[i][None,:]-X2[Z_0[:,i].astype(np.bool)]
        
        for j in range(len(Piv)):
            if diag:
                cov[i]=cov[i]+np.diag(Piv[j]**2)
            else:
                cov[i]=cov[i]+np.reshape(Piv[j],(-1,1))@np.reshape(Piv[j],(1,-1))
        cov[i]=(cov[i])/N_obs+np.eye(p)/100
        

    
    Z=Z_0.copy()#stock log of pivot latent variable
    Error=np.zeros(n_iter)
    Score_list=np.zeros((n_iter,2))
    #Begin EM
    
    print("begin iter")
    for i in range(n_iter):
        #E step
        print("iter",i)
        Label_est=predict_label_with_Z(Z)
        sc=score(Label_est,Label)
        print("score",sc)
        Score_list[i,0]=sc[0]
        Score_list[i,1]=sc[1]
        F1_old=F1.copy()
        F3_old=F3.copy()
        Z,log_pi=compute_Z_pivot(X1,X2,X3,log_pi,F1,mu,cov,F3)# E step
        
        # M Step and M step on mixing weights
        for k in range(nb_cluster):
            N_k=np.exp(logsumexp(Z[:,k]))

            X1_pond=np.exp(Z[:,k])[:,None,None]*X1
            X1_pond_sum=X1_pond.sum(axis=0)/N_k
            F1[k]=vmf.mle(X1_pond_sum)# 

            X2_pond=np.exp(Z[:,k])[:,None]*X2
            mu[k]=X2_pond.sum(axis=0)/N_k
            Piv=mu[k][None,:]-X2
            cov[k]=np.zeros((p,p))
            for j in range(N_obs):
                if diag:
                    cov[k]=cov[k]+np.exp(Z[j,k])*np.diag(Piv[j]**2)

                else:
                    cov[k]=cov[k]+np.exp(Z[j,k])*np.reshape(Piv[j],(-1,1))@np.reshape(Piv[j],(1,-1))
            cov[k]=cov[k]/N_k+np.eye(p)/100

            X3_pond=np.exp(Z[:,k])[:,None,None]*X3
            X3_pond_sum=X3_pond.sum(axis=0)/N_k
            F3[k]=vmf.mle(X3_pond_sum)# faire le mle
        
        
        Error[i]=np.linalg.norm(F1-F1_old)+np.linalg.norm(F3-F3_old)
        if i==0:
            print("end first iter")
           
    # E step and M step on mixing weights
    Z,log_pi=compute_Z_pivot(X1,X2,X3,log_pi,F1,mu,cov,F3)
    
    return log_pi,F1,mu,cov,F3,Z,Error,Score_list

def compute_Z_pivot(X1,X2,X3,log_pi,F1,mu,cov,F3):
    """
    X1,X2,X3 -> the observations
    theta =log_pi,F1,mu,cov,F3 the parameter of the model


    compute the E step, i.e., compute p(Z|X1,X2,X3,theta) stacked in the variable Z,
    the size of Z is Nx nb_cluster with N the number of observations
    
    
    """
    N_obs=len(X1)
    n,p=X1[0].shape
    nb_cluster=len(F1)
    Z=np.zeros((N_obs,nb_cluster,5))

    # we define some normalizing constant seprately to avoid numerical instabilities
    const1=np.zeros(nb_cluster)
    const2=np.zeros(nb_cluster)
    const3=np.zeros(nb_cluster)
    for k in range(nb_cluster):
        const1[k]=spa.log_vmf(F1[k])
        const2[k]=fast_logdet(cov[k])/2+p*np.log(2*np.pi)/2
        const3[k]=spa.log_vmf(F3[k])
    
    const=np.mean(const1)+np.mean(const3)+np.mean(const2)
    const_pi=np.mean(log_pi)
    for j in range(N_obs):

        for k in range(nb_cluster):
            #we compute each attachment terms separetely to avoid logarithm explosion
            Z[j,k,0]=log_pi[k]-const_pi
            Z[j,k,1]=vmf.log_von_mises_fisher(X1[j],F1[k],normalized=False)
            Z[j,k,2]=vmf.log_von_mises_fisher(X3[j],F3[k],normalized=False)
            Z[j,k,3]=log_gaussian(X2[j],mu[k],cov[k])
            Z[j,k,4]=-const1[k]-const2[k]-const3[k]+const
        # we remove the mean to recenter the float
        Z[j]=Z[j]-Z[j].mean(axis=0)
    Z=Z.sum(axis=2)# we sum the contribution
    for j in range(N_obs):
        s=logsumexp(Z[j])
        Z[j]=Z[j]-s
    
    log_pi=logsumexp(Z,axis=0)-np.log(N_obs)
    return Z,log_pi

def score(Label_est,Label_true):
    """
    Compute the accuracy adn the recall related to the predicted Labels
     and the true Labels
    """
    N_obs=len(Label_true)
    if len(Label_true)==len(Label_est):
        sc1=0
        sc2=0
        N1=0
        N2=0
        for i in range(N_obs):
            for j in range(i):
                if Label_true[i]==Label_true[j]:
                    N1+=1
                    sc1=sc1+(Label_est[i]==Label_est[j])*1
                else:
                    N2+=1
                    sc2=sc2+(Label_est[i]!=Label_est[j])*1
        sensitivity=sc1/N1
        specificity=sc2/N2
        print("precision",sc1/(sc1+N2-sc2))
        total_N=N1+N2
        acc,rec=(sc1+sc2)/total_N,sc1/N1
        print("f1 score",2*sc1/(sc1+total_N-sc2))
        return acc,rec
                
    else:
        return "error"










        