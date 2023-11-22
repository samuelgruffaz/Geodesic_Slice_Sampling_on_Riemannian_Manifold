import numpy as np
import scipy.linalg as scl
import matplotlib.pyplot as plt
from numpy.linalg import qr 

ii=complex(0,1)




# We work on the grassman manifold by using a representation of each point on the Stiefel manifold
class Sampler_Slice_Sampling_Grassman():
    

    def __init__(self,log_prob,n,p):
        """
        Parameters:  
            
            log_prob function (nxp-dimensional array-> scalar): log_density on the Grassman manifold .
            n, p(int) : Dimensionparameters for the Grassman manifold. It should be n > p.
        
        """
        self.log_prob=log_prob
        self.n=n
        self.p=p
        self.counts_stepping=None
        self.counts_shrink=None
        self.error_reprojection=[]
        self.rate=None
        self.optimal_rate=0.234 # proposition but not theoric optimal rate
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
    

    def sample_tangent_sphere(self,X):
        '''
        Sample uniformly from the tangent sphere at X.
        An element D of the tangent space at X can be expressed as D =X_perp B where  B is (n-p)xp dimensional
        and X_perp together with X forms an orthonormal basis (ONB) of R^d.
        
        Parameters:  
            
            X (nxp-dimensional array): Point on the Grassman manifold at which the tangent sphere is sampled from.
        
        Returns:
            U,sigma,VT The SVD decomposition of the points sampled on the tangent space at X
            X_V= X x VT

        '''
        #Draw uniformly from the sphere
        n=self.n
        p=self.p
        dim_grassman = int( p*(n-p))
        raw_samples = np.random.normal(size = dim_grassman)
        raw_samples = raw_samples / (np.linalg.norm(raw_samples) + + 1e-100)
    
    

        #Use the samples to make matrix B. Compute SVDof X_perp B.
        X_perp = self.null_space(X)
        B = np.reshape(raw_samples, (n-p, p))
        H=X_perp.dot(B)
        U, sigma, VT=np.linalg.svd(H, full_matrices=False)
        
        X_V=X.dot(VT.T)
        
        return (X_V,U,sigma,VT)
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
    def proj_V(self,X):
        """Orthogonal projection onto V(n,p) for a matrix X with full rank"""
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

        #Reorthogonalise before returning the new point on the Grassman manifold.
        # We do this to counteract the numerical error induced in each stepp.
        # If we do not do this we will already after two steps "leave" the Grassman manifold.
        if reproject:
            Reprojection= self.proj_V(Y)
            self.error_reprojection.append(scl.norm(Reprojection-Y))
        else:
            Reprojection=Y
        # we compute the error of reprojection at each step, we can observe this to inspect instabilities
        
        return Reprojection
    


    def stepping_out(self,w,Input, level= None, log_density= None, m = 1):
        '''
        Stepping-out procedure.
        
        Parameters:
            log_density (function) : Should take arguments from the space where the stepping out happens and return real numbers . Levelsets of this function are targeted by the stepping-out procedure. Is not needed if m=1.
            Input (triplet) : (X,V,to_exp) quantities to compute geodesics
            level (float) : Level belonging to the level set targeted by the stepping-out procedure. Should be > 0. Is not needed if m = 1.
            w (float) : Length of the stepping out interval.
            m (int) : Maximum number of stepping out steps. Default 1.
            
        Returns:
            a,b (float) : Determine the interval which results from the stepping out procedure.
        '''
        
        
        a = - np.random.uniform(0,w)
        b = a + w
        
        if m == 1:
            return (a,b)
            
        J = np.random.randint(0,m)
        # J_pythoncode = J_pseudocode - 1
        # Stepping out to the left.
        i = 1
        count=0
        gamma_a=self.walk_geodesic(Input, a,reproject=False)
        
        while i <= J and log_density(gamma_a) > level:
            
            a = a - w
            i = i + 1
            count+=1
            gamma_a=self.walk_geodesic(Input, a,reproject=False)
        
        self.counts_stepping[self.nn]=count 
        
        # Stepping out to the right.
        i = 1
        gamma_b=self.walk_geodesic(Input, b,reproject=False)
        while i <= m-1-J and log_density(gamma_b) > level:
            b = b + w
            i = i+1
            gamma_b=self.walk_geodesic(Input, b,reproject=False)
        
        
        return (a,b)
        
        
    def shrinkage(self,a,b,Input, level, log_density):  
        '''
        Shrinkage procedure.
        
        Parameters:
            a,b (float) : Upper and lower bound for the interval to be shrinked.
            log_density (function) : Should take arguments from the space where the shrinkage happens and return real numbers. Levelsets of this function are targeted by the shrinkage procedure.
            Input (triplet) : (X,V,to_exp) quantities to compute geodesics
            level (float) : Level belonging to the level set targeted by the shrinkage procedure. Should be between 0 and density(walker(0)).
            
        Returns:
            y : Point on the samping space obtained by the shrinkage procedure
        '''
        
        
        theta = np.random.uniform(0, b-a)
        theta_h=theta+(a-b)*(theta>b)
        theta_min = theta
        theta_max = theta
        
        y = self.walk_geodesic(Input, theta_h,reproject=False)
        
        count=1
        while log_density(y) <= level and count<100:
            
            count+=1
            if theta> theta_min and theta<b-a:
                theta_min = theta
            else:
                theta_max = theta
            theta1 = np.random.uniform(0, theta_max)
            theta2 = np.random.uniform(theta_min, b-a)
            uu = np.random.uniform(0, 1)
            p=theta_max/(b-a-theta_min+theta_max)
            
            if uu<p:
                theta=theta1
            else:
                theta=theta2
            theta_h=theta+(a-b)*(theta>b)
            y = self.walk_geodesic(Input, theta_h,reproject=False)
        y1=self.proj_V(y)
        self.error_reprojection.append(scl.norm(y1-y1))
        self.counts_shrink[self.nn]=count
        self.chosen_theta[self.nn]=abs(theta)
        return y1,count
        
        
    def GSS_Grassman(self,log_density,X,w, m=1):
        '''
        Transition kernel of the Geodesic slice sampler (GSS) for the Grassman manifold.
        
        Parameters:
            log_density (function): Takes nxp-dimensional arrays (= points on the Grassman manifold) and returns real numbers . Describes invariant distribution of the GSS.
            X (nxp-dimensional array) : Current point on the Grassman manifold.
            n,p (int) : Dimensionparameters for the Grassman manifold. It should be n > p.
            w (float) : Interval lenght for the stepping-out procedure.
            m (int) : Maximum number of stepping-out steps. Default m = 1.
            
        Returns: 
            (nxp-dimensional array) : Point on the Grassman manifold obtained by one step of the GSS.
        '''
        
        level = np.log(np.random.uniform(0, 1))+log_density(X)
        Input = self.sample_tangent_sphere(X)
        a,b = self.stepping_out(w,Input, level=level, log_density=log_density, m=m)
        Y = self.shrinkage(a,b,Input, level=level, log_density=log_density)
        
        return Y
        
    def run_kernel(self,X_0, n_iter,w=1,m=1,log_density=None,use_adapt=False):
        '''
        Simulates the trajectory of a Markov chain.
        
        Parameters:
            X_0 : Initial point.
            n_iter (int) : Lenght of the trajectory.
            log_density: to redifine the log_density if None, we use the definition in init
            use_adapt: False to not adapt the value of w
            
        Returns:
            data (list) : Contains the samples on the Grassman manifold n,p related to the log_desnity.
            rate : the average ratio of acceptation per step, 1/count with count the number of trial in the shrinkage
        ''' 
        if log_density is None:
            log_density=self.log_prob
        self.counts_shrink=np.zeros(n_iter)
        self.counts_stepping=np.zeros(n_iter)
        self.chosen_theta=np.zeros(n_iter)
        self.nn=0
        data = [X_0]
        prop_w=w
    
        X=X_0
        rate_sum=0
        for i in range(n_iter//20):# we devide by 20 to loop on 20 to estimate the "acceptance ratio"
            
            sum=0
            for j in range(20):
               
                X,count = self.GSS_Grassman(log_density, X, prop_w,m=m)
                sum=sum+1/count# we devide by the number of computation, count =1->acceptance ratio=1
                self.nn=self.nn+1
                data.append(X.copy())
            rate_X=sum/20
            rate_sum+=sum
            if use_adapt:#adaptative step
                adaptive_X = 2*(rate_X > self.optimal_rate)-1
                prop_w = np.exp(np.log(prop_w) + 0.5*adaptive_X/(1+i)**0.6)
        rate=rate_sum/n_iter
        print("prop_w",prop_w)
        print("rate_X",rate_X)
        self.rate=rate
        return data,rate
    
