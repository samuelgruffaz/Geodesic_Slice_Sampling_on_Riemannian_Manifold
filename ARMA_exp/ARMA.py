import numpy as np

class ARMA():
    """
    Implement a model ARMA
    
    """
    def __init__(self,R,Q,B,F,p,n):
        self.R=R # Covariance matrix for the observations noise
        self.Q=Q # Covariance matrix for the latent noise
        self.B=B # Transition matrix on latent
        self.F=F # Mises-Fisher distribution parameter for the Prior
        self.n=n # Dimension of the observations
        self.p=p # Dimension of the latent
        
    def proj_V(self,X):
        """Orthogonal projection onto V(n,p) for a matrix X with full rank"""
        u, _, v = np.linalg.svd(X, full_matrices=False)
        return u@v
    def log_prob(self,Z):
        """
        take the vectors of observations (z_i) and return the log-posterior function
        """
        prior_H = lambda x : np.einsum('ik,ik->k',self.F,x).sum()
        N=len(Z)
        def posterior(H):

            l=0
            y_tilde=Z[0]#reducteur
            P_kkminus=self.Q#reducteur
            S=H@(P_kkminus@H.T)+self.R
            S_inverse=np.linalg.inv(S)
            K=P_kkminus@(H.T@S_inverse)
            x_k=K.dot(y_tilde)#reducteur
            P_kk=(np.eye(self.p)-K@H)@P_kkminus
            for i in range(N): # Kalman iterations to compute the attachment terms
                #https://en.wikipedia.org/wiki/Kalman_filter
                l=l-(y_tilde@(S_inverse@ y_tilde)+np.log(np.linalg.det(S)))
                if i<N-1:
                    x_pred=self.B.dot(x_k)
                    P_kkminus=self.B@(P_kk@self.B.T)+self.Q
                    y_tilde=Z[i+1]-H.dot(x_pred)
                    
                    S=H@(P_kkminus@H.T)+self.R
                    S_inverse=np.linalg.inv(S)
                    K=P_kkminus@(H.T@S_inverse)
                    x_k=x_pred+K.dot(y_tilde)#reducteur
                    P_kk=(np.eye(self.p)-K@H)@P_kkminus



            return l+prior_H(H)

        return posterior
    
    def generator(self,N,H):
        """
        Take the number of observations N ans the observations matrix H to generate
        N observations.
        
        """
        x_1=np.random.multivariate_normal(np.zeros(self.p),self.Q)
        x_cu=x_1.copy()
        X=np.zeros((N,self.p))
        X[0]=x_1
        Z=np.zeros((N,self.n))
        for i in range(N):
            noise_w=np.random.multivariate_normal(np.zeros(self.n),self.R)
            noise_v=np.random.multivariate_normal(np.zeros(self.p),self.Q)
            Z[i]=H@x_cu+noise_w
            x_cu=self.B@x_1+noise_v
            if i<N-1:
                X[i+1]=x_cu
        return X,Z
    