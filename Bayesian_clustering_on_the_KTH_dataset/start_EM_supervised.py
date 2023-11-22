import os
import numpy as np
import pickle
from clustering_utils import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# This script should be run first


path_save = "Bayesian_clustering_on_the_KTH_dataset/results"
try:
    os.mkdir(path_save)
except:
    pass

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


d_latent2=50
for env in ["d1","d2","d3","d4"]:
    print(env)
    D={}
    print("open data")
    with open(os.path.join(path_features,str(d_latent)+env+".pkl"),'rb') as f:
        (features_list_U,features_list_Sigma,features_list_A,Label_true)=pickle.load(f)
    features_list_U,features_list_Sigma,features_list_A=features_list_U[:,:,:d_latent2],features_list_Sigma[:,:d_latent2],features_list_A[:,:d_latent2,:d_latent2]
    
    Label_true=Label_to_int(Label_true)

    # split the indexes in a train and test set
    X_train, X_test, Label_train, Label_test,ind_train,ind_test = train_test_split(features_list_U, Label_true,np.arange(len(Label_true)), test_size=0.33, random_state=seed)

    # select the features according to the indexes
    features_list_U,features_list_Sigma,features_list_A=features_list_U[:,:,:d_latent2],features_list_Sigma[:,:d_latent2],features_list_A[:,:d_latent2,:d_latent2]
    features_train_U,features_train_Sigma,features_train_A=features_list_U[ind_train],features_list_Sigma[ind_train],features_list_A[ind_train]
    features_test_U,features_test_Sigma,features_test_A=features_list_U[ind_test],features_list_Sigma[ind_test],features_list_A[ind_test]
    
    kmeans_b=True
    diag_b=True
    init=Label_train # on the train set we know the labels to initialize the EM algorithm
    
    print("EM estimation")
    log_pi,F1,mu,cov,F3,Z,Error,Score_list=prod_EM_estimation(features_train_U,features_train_Sigma,features_train_A,Label_train,nb_cluster=N_cluster,n_iter=n_iter,kmens=kmeans_b,diag=diag_b,init=Label_train)

    #score on the train set
    Label_train_pred=predict_label_with_Z(Z)
    print("score",score(Label_train_pred,Label_train))
    D[env+"_score"]=score(Label_train_pred,Label_true)[0]

    #Score on the test set
    Z_test,log_pis=compute_Z_pivot(features_test_U,features_test_Sigma,features_test_A,log_pi,F1,mu,cov,F3)
    Label_test_pred=predict_label_with_Z(Z_test)
    print("accuracy", accuracy_score(Label_test_pred,Label_test))
    print("f1 score weoghted",f1_score(Label_test_pred, Label_test,average="weighted"))

    # Save parameters adn restuls
    tup=(log_pi,F1,mu,cov,F3,Z,Z_test,ind_train,ind_test)
    print("save")
    with open(os.path.join(path_save,'supervised_diag_latent'+str(d_latent2)+'_niter'+str(n_iter)+"clustering_est_"+env+".pkl"),'wb') as f:
        pickle.dump(tup,f)

    





