import numpy as np
import scipy.linalg as scl
import matplotlib.pyplot as plt
from numpy.linalg import qr 

ii=complex(0,1)


class Hybrid_Monte_Carlo_Grassman():
    def __init__(self,log_prob,grad_log_prob,n,p,T,h):
        """
        Parameters:  
            
            log_prob function (nxp-dimensional array-> scalar): log_density on the Grassman manifold .
            n, p(int) : Dimension parameters for the Grassman manifold. It should be n > p.
        
        """
        self.log_prob=log_prob
        self.grad_log_prob=grad_log_prob
        self.n=n
        self.p=p
        self.T=T
        self.h=h
    def null_space(self,X):
        """
        Input :
        X point on the Grassman n,p manifold 
        output: 
         X_perp (n,n-p) array such that X,X_perp is a orthonormal basis of R^n
        """
        u, s, vh = np.linalg.svd(X, full_matrices=True)
        #print(u,s,vh)
        n, p = X.shape
        num=p
        X_perp = u[:,num:]
        
        return X_perp
    def proposal(self,X,v):
        """
        MALA poposal

        Input :
            -X Sitefel representent of a point on the Grassman manifold 
            -velocity sampled on the tangent space at X
        return: 
            X_new, after one step of HMC (in the experiments T=1)
        
        """
        X_new=X
        for i in range(self.T):
            v=self.proj_tan_grassman(X_new,v+self.h*self.grad_log_prob(X_new)/2)#gradient step
            Input=self.to_input(X_new,v)#wrapping for the geodesics
            X_new=self.walk_geodesic(Input,self.h)# geodesics computations
            v=self.proj_tan_grassman(X_new,v+self.h*self.grad_log_prob(X_new)/2)#projection of the velocity 
        X_new=self.proj_V(X_new)
        return X_new,v

    def sample_one(self,X,last_log):
        """
        Input:
            X the current Grassman n,p point
            prop_X a positive scalar to adapt size of the step
        l   ast_log scalar, the log target related to X
        Output:
            X_new the new point in the chain
            last_log the log target related to X_new
            accept =0 if X_new=X  else accept=1 
        
        """
        n,p=self.n,self.p
        
        #sample the velocity with normal distribution
        v=np.random.normal(size = n*p)
        v=np.reshape(v,(n,p))
        #projection on the tangent space at X
        v=self.proj_tan_grassman(X,v)

        #geoMALA proposal
        X2,v_new=self.proposal(X,v)

        #Metropolis steps

        new_log=self.log_prob(X2)#target estimation
        log_alpha = new_log -np.linalg.norm(v_new,2)**2/2- last_log+np.linalg.norm(v,2)**2/2
            # [X] Accept or reject
        if np.log(np.random.rand()) < log_alpha:# acceptance step
            X_new = X2
            last_log = new_log
            accept = 1
        else:
            X_new = X
            accept = 0
        return X_new,last_log,accept

    def log_grass(self,X,Y):
        """
        Logarithm of Y at X on the Grassman manifold
        """
        U,sigma,VT=np.linalg.svd((Y.T).dot(X),full_matrices=False)
        Y_star=Y.dot(U.dot(VT))
        U_2,sigma_2,VT_2=np.linalg.svd((np.eye(len(X))-X.dot(X.T)).dot(Y_star),full_matrices=False)
        Sigma=np.arcsin(sigma_2)
        Delta=(U_2*Sigma)@VT_2
        return Delta

    def proj_tan_grassman(self,X,V):
        """
        Projection of V on the tangent space at X in the grassman manifold
        
        """
        New_V=V-(X.dot(X.T)).dot(V)
        return New_V

    def proj_V(self,X):
        """Orthogonal projection onto V(n,p) (Stifel) for a matrix X with full rank"""
        u, _, v = np.linalg.svd(X, full_matrices=False)
        return u@v
    def to_input(self,X,H):
        """ wrapper function for the geodesics, (see Remark 8 in the paper)
        Input :
            X a stiefel representant of a point on a point on the grassman manifold
            H a point on the tangent space at X on the grassman manifold
        Return : 

            U,sigma,VT the svd of H
            X_V= X x VT
        """    
        U, sigma, VT=np.linalg.svd(H, full_matrices=False)
        
        X_V=X.dot(VT.T)
        
        return (X_V,U,sigma,VT)

    def walk_geodesic(self,Input,t,reproject=True):
        """
        Input=(X_V,U,sigma,VT), t the time,
         with U,sigma,VT the svd of a point on the tangent space
          and X_V=XVT with X the starting point of the geodesic
          return Reprojection=Gamma_(X,UsigmaVT)(t) (see Remark 8 in the paper)
          
        
        """


        X_V,U,sigma,VT=Input
        
        rec_sigma=np.concatenate([np.diag(np.cos(sigma*t)),np.diag(np.sin(sigma*t))], axis=0)
        Y=(np.concatenate([X_V,U], axis=1)@rec_sigma).dot(VT)

        if reproject:# reprojection step to avoid accumulation of numerical errors
            Reprojection= self.proj_V(Y)
        else:
            Reprojection=Y

        return Reprojection
    def run_kernel(self,X_0,n_iter):
        """
        GeoMALA sampling beginning from X_0
        Input:
            - X_0 the initial Stiefel point
            - n_iter the number of samples
        return:
            the n_iter samples on the grassman manifold represented on the stiefel
        
        """
        last_log=self.log_prob(X_0)
        samples=[X_0]
        X_new=X_0
        for i in range(n_iter//20):
            sum=0
            for j in range(20):
                X_new,last_log,accept=self.sample_one(X_new,last_log)
                samples.append(X_new.copy())
                sum=sum+accept
            
            rate_X=sum/20
            
        print("rate_X",rate_X)#acceptation rate
        
        return samples
      

    
