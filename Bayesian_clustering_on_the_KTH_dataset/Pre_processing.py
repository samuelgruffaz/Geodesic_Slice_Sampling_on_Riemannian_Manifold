import os
import numpy as np
import glob
from avi_r import AVIReader
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage.transform import resize
import math
from skimage.feature import hog
import pickle

     
d=3780

#TODO to fill
path_data="TOFILL"
path_to_save_features="TOFILL"

d_latent=200
Actions=["walking","boxing","handclapping","handwaving","jogging","running"]
persons=["person0"+str(i) for i in range(1,10)]+["person"+str(i) for i in range(10,26)]
print(persons)

def feature_video(video,d_latent=50):
    """
    function to compute the ARMA features related to the video with d_latent dimension
    in the latent space
    """
    L=[]
    for frame in video:
        # frame is a avi_r.frame.Frame object
        #image = frame.numpy()
        
        resized_img = resize(frame.numpy(), (64, 128))
        
        fd= hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=False,multichannel=True)
        d=len(fd)
        L.append(fd)
    video.close()
    
    T=len(L)
    vec=np.zeros((d,T))
    for i in range(T):
        vec[:,i]=L[i]
    U, Sigma, V = np.linalg.svd(vec, full_matrices=False)
    U_n=U[:,:d_latent]
    Sigma_n=Sigma[:d_latent]
    V_n=V[:d_latent,:]
    
    D_1=np.zeros((T,T))
    D_1[1:,:T-1]=np.eye(T-1)
    D_2=np.zeros((T,T))
    D_2[:T-1,:T-1]=np.eye(T-1)
    A=np.diag(Sigma_n)@V_n@D_1@V_n.T@np.linalg.inv(V_n@D_2@V_n.T)@np.diag(1/Sigma_n)
    return U_n,Sigma_n,A
def feature_environement(env,d_latent=50):
    #Function to preprocess the videos related to an environement,
    #We use AVIReader
    if env=="d3":
        N=25*6-1
    else:
        N=25*6
    features_list_U=np.zeros((N,d,d_latent))
    features_list_Sigma=np.zeros((N,d_latent))
    features_list_A=np.zeros((N,d_latent,d_latent))
    Label=[]
    list_path=[]
    for i in range(6):
        for j in range(25):
            Label.append(Actions[i])
            list_path.append(os.path.join(path_data,Actions[i],persons[j]+'_'+Actions[i]+'_'+env+'_uncomp.avi'))
            if not(os.path.exists(list_path[-1])):
                print(env,Actions[i],persons[j])
                e=list_path.pop()
                e=Label.pop()
                print("last",list_path[-1])
    for i in range(len(list_path)):
        
        filename=list_path[i]
        print(filename)
        
        video_ = AVIReader(filename)
        U,Sigma,A=feature_video(video_,d_latent=d_latent)
        features_list_U[i]=U
        features_list_Sigma[i]=Sigma
        features_list_A[i]=A
        print("done")
    return features_list_U,features_list_Sigma,features_list_A,Label



# We apply the previous function to all videos in each environement and save the fetautres as numpy array
#the only parameter is d_latent
for env in ["d1","d2","d3","d4"]:
    print(env)
    features_list_U,features_list_Sigma,features_list_A,Label=feature_environement(env,d_latent=d_latent)
    to_save=(features_list_U,features_list_Sigma,features_list_A,Label)
    with open(os.path.join(path_to_save_features,str(d_latent)+env+".pkl"),'wb') as f:
        pickle.dump(to_save,f)



