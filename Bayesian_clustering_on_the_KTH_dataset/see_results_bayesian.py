import os
import numpy as np
import pickle
from clustering_utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


#This script presents the results if you have the data 


path_save = "Bayesian_clustering_on_the_KTH_dataset/results"
seed=22
np.random.seed(seed)

d=3780 # dimension of the observations space
N_cluster=6 # number of cluster
n_iter=2 # number of EM iterations
d_latent=200 # dimension of the latent space
d_latent2=50 # dimension of the latent space reduced, to reduce the computation time

path_features="path where the features are saved after the preprocessing"

Actions=["walking","boxing","handclapping","handwaving","jogging","running"]
persons=["person0"+str(i) for i in range(1,10)]+["person"+str(i) for i in range(10,26)]


def Label_to_int(labels):
    D={}
    for i in range(len(Actions)):
        D[Actions[i]]=i
    labels_new=np.zeros(len(labels))
    for i in range(len(labels)):
        labels_new[i]=int(D[labels[i]])
    return labels_new

def analyze_std(weights,labels_std,Label_test):
    """
    function used to
    vMF clustering MCMC lower variance
    from the cluster assignments weights (weights) , the standard deviation of the related labels (labels_std)
    and the true labels (Label_test)
    """
    
    w_mean=weights.mean(axis=0)
    
    #find the observations indices where the variance of the labels is 0
    G=[i for i in range(len(labels_std)) if labels_std[i]==0]
    #print the number of points removed
    print("nb change",len(labels_std)-len(G))
    # compute the mean assigments on this restricted dataset
    weights_without_top_var=w_mean[G,:]
    #make the final prediction on this restricted dataset
    Label_est_var=predict_label_with_Z(weights_without_top_var)
    print("score vMF clustering MCMC lower variance")
    
    # print the related score
    print("accuracy", accuracy_score(Label_est_var,Label_test[G]))
    print("f1 score weoghted",f1_score(Label_est_var, Label_test[G],average="weighted"))

    return G
d_latent=200

d_latent2=50
for env in ["d1","d2","d3","d4"]:
    print(env)
    D={}
    print("open data")
    with open(os.path.join(path_features,str(d_latent)+env+".pkl"),'rb') as f:
        (features_list_U,features_list_Sigma,features_list_A,Label_true)=pickle.load(f)
    features_list_U,features_list_Sigma,features_list_A=features_list_U[:,:,:d_latent2],features_list_Sigma[:,:d_latent2],features_list_A[:,:d_latent2,:d_latent2]
    
    Label_true=Label_to_int(Label_true)
    X_train, X_test, Label_train, Label_test,ind_train,ind_test = train_test_split(features_list_U, Label_true,np.arange(len(Label_true)), test_size=0.33, random_state=seed)
    with open(os.path.join(path_features,str(d_latent)+env+".pkl"),'rb') as f:
        (features_list_U,features_list_Sigma,features_list_A,Label_true)=pickle.load(f)

    print("open")
    with open(os.path.join(path_save,'supervised_diag_latent'+str(d_latent2)+'_niter'+str(n_iter)+"clustering_est_"+env+".pkl"),'rb') as f:
        tup=pickle.load(f)
    (log_pi,F1,mu,cov,F3,Z,Z_test,ind_train2,ind_test2)=tup
    
    
    Label_test_pred=predict_label_with_Z(Z_test)

    # the score with only EM
    print("score vMF clustering EM")
    print("accuracy", accuracy_score(Label_test_pred,Label_test))
    print("f1 score weoghted",f1_score(Label_test_pred, Label_test,average="weighted"))

    kmeans_b=True
    diag_b=True
    init=Label_true
    init=None

    with open(os.path.join(path_save,'weights_supervised_diag_latent'+str(d_latent2)+'_niter'+str(n_iter)+"clustering_est_"+env+".pkl"),'rb') as f:
        weights=pickle.load(f)

    print(weights.shape)

    # computes the labels according to all the different weights samples
    Label_est_big=np.zeros((len(weights),len(Label_test)))
    for i in range(len(Label_est_big)):
        Label_est_big[i]=predict_label_with_Z(weights[i])
    
    
    weights_mean=np.exp(weights.mean(axis=0))
    Label_est_new=predict_label_with_Z(weights_mean)

    # estimation with the mean of the weights assigments
    print("score vMF clustering MCMC")
    print("accuracy", accuracy_score(Label_est_new,Label_test))
    print("f1 score weoghted",f1_score(Label_est_new, Label_test,average="weighted"))
   #vMF clustering MCMC lower variance
    G=analyze_std(weights,Label_est_big.std(axis=0),Label_test)
    

