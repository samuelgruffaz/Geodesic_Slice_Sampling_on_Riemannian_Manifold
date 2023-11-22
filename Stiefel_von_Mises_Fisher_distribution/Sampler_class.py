import numpy as np
import scipy.linalg as scl
from numpy.linalg import qr 

ii=complex(0,1)



class Sampler_geo_mh():

    def __init__(self,log_prob,n,p):
        """
        n is the number of dimensions
        p the number of vectors related to the Stiefel n,p
        log_prob is a function taking a point on the Stiefel n,p and returning a scalar,
        it corresponds to the log_density
        
        """
        self.log_prob=log_prob
        self.n=n
        self.p=p
        self.optimal_rate=0.234 # we choose this constant to maximize the asymptotic speed of mixing, 
        # optimal design result for mcmc chain
    def sample_one(self,X,prop_X,last_log):
        """
        Input:
        X the current Stiefel n,p point
        prop_X a positive scalar to adapt size of the step
        last_log scalar, the log target related to X
        Output:
        X_new the new point in the chain
        last_log the log target related to X_new
        accept =0 if X_new=X  else accept=1 
        
        """
        X2=self.proposal(X,prop_X)# proposal
        #target estimation
        new_log=self.log_prob(X2)
        log_alpha = new_log - last_log
            # [X] Accept or reject
        if np.log(np.random.rand()) < log_alpha:# acceptance step
            X_new = X2
            last_log = new_log
            accept = 1
        else:
            X_new = X
            accept = 0
        return X_new,last_log,accept

    def run_kernel(self,X_0,n_iter,prop_X=0.01):
        """
        
        Input:
        X_0 the initial Stiefel point
        n_iter the number of samples
        prop_X the inital size of steps 
        
        """
        last_log=self.log_prob(X_0)
        samples=[X_0]
        X_new=X_0
        for i in range(n_iter//20):
            sum=0
            for j in range(20):
                X_new,last_log,accept=self.sample_one(X_new,prop_X,last_log)
                samples.append(X_new.copy())
                sum=sum+accept
            #estimation of the acceptance rate
            rate_X=sum/20
            # adaptative procedure to target the optimal rate
            adaptive_X = 2*(rate_X > self.optimal_rate)-1
            prop_X = np.exp(np.log(prop_X) + 0.5*adaptive_X/(1+i)**0.6)
        print("rate_X",rate_X)
        print("prop_X",prop_X)
        return samples

    def null_space(self,X):
        """
        Input :
        X point on the Stiefel n,p manifold 
        output: 
         X_perp (n,n-p) array such that X,X_perp is a orthonormal basis of R^n
        """
        u, s, vh = np.linalg.svd(X, full_matrices=True)
        #print(u,s,vh)
        n, p = X.shape
        num=p
        X_perp = u[:,num:]
        
        return X_perp
    def pre_walk_geodesic(self, V):
        '''
        
        Compute some quantities useful to compute geodesics on the stiefel manifold
        we should compute the exponential map of an antisymetric matrices,
        we use the eigen values decompistion for complex matrices
        
        Parameters:

            V ((pxp-dimensional array, dxd-dimensional array, dxk-dimensional array)) : Specifies elemnt of the tangent sphere at X that gives direction of the geodesic.
            Should be of the form (A,Q,R) such that the element of the tangent sphere is given by XA+QR
            where A is skew-symmetric, Q is orthogonal and R an upper-triangular matrix.
        
            
        Returns:
            the svd decomposition of i x arg: v_l,diag_e, v_r' since it is autoadjoint in the complex matrice space
            and v_r is v_r' x vec where vec is a constant vector implied in the geodesics computations
        '''
        n=self.n#d, k (int) : Dimensionparameters for the Stiefel manifold. It should be d > k.
        p=self.p
        # Compute XM + QN, where
        # M = exp[t (A  -R^T)] * Id_k
        # N      [  (R     0)]     0 .
        A,Q,R = V
        arg_top = np.hstack((A, -np.transpose(R)))
        arg_bottom = np.hstack((R, np.zeros((p,p))))
        arg = np.vstack((arg_top, arg_bottom))
        
        vec = np.vstack((np.eye(p), np.zeros((p,p))))
        v_l,diag_e,v_r=self.pre_exp(arg,vec)
        
        return v_l,diag_e,v_r

    def walk_geodesic(self,Input,t):
        """
        Take Input=(X,V,to_exp) with X the Stiefel point, V its velocity in the TM_X and to_exp some quantities
        to compute the geodesics at time t
        (we reproject to avoid numerical errors, we can accelerate the codes by not reprojecting at each step)
        
        """
        X,V,to_exp=Input
        v_l,diag_e,v_r=to_exp
        A,Q,R = V
        n,p=X.shape
        H=self.expm(v_l,diag_e,v_r,t)
        M, N = np.split(H, [p])
        Y = X.dot(M) + Q.dot(N)
         #Reorthogonalise before returning the new point on the Stiefel manifold.
        # We do this to counteract the numerical error induced in each stepp.
        # If we do not do this we will already after two steps "leave" the Stiefel manifold.
    
        Reprojection= self.proj_V(Y)
        # we compute the error of reprojection at each step, we can observe this to inspect instabilities
        
        return Reprojection
    def pre_exp(self,M,vec):
        """
        we compute the complex eigenvalues of iM which is autoadjoint as a complex matrix
        and we precompute the product of basis of decomposition with vec
        """
        e,v=np.linalg.eigh(M*ii)
        return v,e,(v.T.conj())@vec
    def expm(self,v_l,e,v_r,t):
        """ 
        this formula combines with pre_exp gives nearly the same result than scipy.expm,
        we use the antisymetric structure to accelerate the computation
        """
        return np.real(v_l@np.diag(np.exp(-t*e*ii))@v_r)

    def sample_tangent_sphere(self,X):
        '''
        Sample uniformly from the tangent sphere at X.
        An element D of the tangent space at X can be expressed as D = XA + X_perp B where A is pxp-dimensional and skew symmetic, B is (n-p)xp dimensional
        and X_perp together with X forms an orthonormal basis (ONB) of R^d.
        
        Parameters:  
            
            X (nxp-dimensional array): Point on the Stiefel manifold at which the tangent sphere is sampled from.
            n, p(int) : Dimensionparameters for the Stiefel manifold. It should be n > p.
        
        Returns:
            V=(A,Q,R) where
            A (pxp-dimensional array)
            Q (nxn -dimensional array) : Orthogonal matrix from the QR-decomposition of X_perp B
            R (nxp-dimensional array) : Upper triangular matrix from the QR-decomposition of X_perp  
            and to_exp (v_l,diag_e,v_r) where
                v_l f(diag_e,t)v_r enable us to compute the geodesic at time t with f a well chosen function
                describe in expm 

        '''
        #Draw uniformly from the sphere
        n=self.n
        p=self.p
        dim_stiefel = int(p*(p-1)/2 + p*(n-p))
        raw_samples = np.random.normal(size = dim_stiefel)
        raw_samples = raw_samples 
    
        
        #Map sample to the tangent space of the Stiefel manifold at X.
        # Use the first p(p-1)/2 samples to make the skew-symmetric matrix A.
        A = np.zeros((p,p))
        ind = np.triu_indices(p, 1)
        A[ind] = raw_samples[:int(p*(p-1)/2)]
        A = A - np.transpose(A)
    

        #Use the remaining samples to make matrix B. Compute QR-decomposition of X_perp B.
        X_perp = self.null_space(X)
        B = np.reshape(raw_samples[int(p*(p-1)/2):], (n-p, p))
        Q,R = qr(X_perp.dot(B))
        V=(A,Q,R)

        # compute quantities to estimate the geodesics starting from X with velocity V
        # we use the antisymetric structure of V
        to_exp=self.pre_walk_geodesic(V)
        
        return V,to_exp

    def proposal(self,X,prop_X):
        """
        Input: 
        X the Stiefel point n,p
        propX a positve scalar to adapt the step
        Output:
        the proposal on the Stiefel X2
        """
        V,to_exp=self.sample_tangent_sphere(X)
        Input=(X,V,to_exp)
        X2 = self.walk_geodesic(Input,prop_X)
        
        return X2

    def proj_V(self,X):
        """Orthogonal projection onto V(n,p) for a matrix X with full rank"""
        u, _, v = np.linalg.svd(X, full_matrices=False)
        return u@v


class Sampler_mh():
    """
    The adaptative Metropolis Hasting sampler with reprojections
    """
    def __init__(self,log_prob,n,p):
        """
        n is the number of dimensions
        p the number of vectors related to the Stiefel n,p
        log_prob is a function taking a point on the Stiefel n,p and returning a scalar,
        it corresponds to the log_density
        
        """
        self.log_prob=log_prob
        self.n=n
        self.p=p
        self.optimal_rate=0.234 # we choose this constant to maximize the asymptotic speed of mixing, 
        # optimal design result for mcmc chain
    def sample_one(self,X,prop_X,last_log):
        """
        Input:
        X the current Stiefel n,p point
        prop_X a positive scalar to adapt size of the step
        last_log scalar, the log target related to X
        Output:
        X_new the new point in the chain
        last_log the log target related to X_new
        accept =0 if X_new=X  else accept=1 
        
        """
        X2=self.proposal(X,prop_X)# proposal
        #target estimation
        new_log=self.log_prob(X2)
        log_alpha = new_log - last_log
            # [X] Accept or reject
        if np.log(np.random.rand()) < log_alpha:# acceptance step
            X_new = X2
            last_log = new_log
            accept = 1
        else:
            X_new = X
            accept = 0
        return X_new,last_log,accept

    def run_kernel(self,X_0,n_iter,prop_X=1):
        """
        
        Input:
        X_0 the initial Stiefel point
        n_iter the number of samples
        prop_X the inital size of steps 
        
        """
        last_log=self.log_prob(X_0)
        samples=[X_0]
        X_new=X_0
        for i in range(n_iter//20):
            sum=0
            for j in range(20):
                X_new,last_log,accept=self.sample_one(X_new,prop_X,last_log)
                samples.append(X_new.copy())
                sum=sum+accept
            #estimation of the acceptance rate
            rate_X=sum/20
            # adaptative procedure to target the optimal rate
            adaptive_X = 2*(rate_X > self.optimal_rate)-1
            prop_X = np.exp(np.log(prop_X) + 0.5*adaptive_X/(1+i)**0.6)
        print("rate_X",rate_X)
        print("prop_X",prop_X)
        return samples

    def proposal(self,X,prop_X):
        """
        Input: 
        X the Stiefel point n,p
        propX a positve scalar to adapt the step
        Output:
        the proposal on the Stiefel X2
        """
        D = prop_X*np.random.randn(self.n,self.p)
        X2 = self.proj_V(X + D)# Reprojection is really necessary for large prop_X,n,p
        return X2

    def proj_V(self,X):
        """Orthogonal projection onto V(n,p) for a matrix X with full rank"""
        u, _, v = np.linalg.svd(X, full_matrices=False)
        return u@v

class Sampler_Slice_Sampling_Straight_lines():

    def __init__(self,log_prob):
        """
        Log_prob is the log density fonction of the density to sample in R^d
        """
        self.log_prob=log_prob
       
        self.counts_stepping=None
        self.counts_shrink=None

    def walk_geodesic(self,input,a):
        """
        In R^d with the euclidean metric, geodesics are straight lines
        input: 
        input=(x,v) in (R^d)^2, x the point and v the velocity
        a, a scalar representign the time to walk on the geodesic

        output: 
        the geodesic at time a beginning from x with velocity v 
        
        """
        x,v=input
        y=x+a*v
        return y
    def stepping_out(self,w,Input, level= None, log_density= None, m = 1):
        '''
        Stepping-out procedure.
        
        Parameters:
            log_density (function) : Should take arguments from the space where the stepping out happens and return real numbers. Levelsets of this function are targeted by the stepping-out procedure. Is not needed if m=1.
            walker (function) : Should take float arguments and return points that can be passed to denisty. Used to make the stepping out steps. Is not needed if m=1.
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
        
        # Stepping out to the left.
        i = 1
        count=0
        gamma_a=self.walk_geodesic(Input, a)
        
        while i <= J and log_density(gamma_a) > level:
            
            a = a - w
            i = i + 1
            count+=1
            gamma_a=self.walk_geodesic(Input, a)
        
        self.counts_stepping[self.nn]=count 
        
        # Stepping out to the right.
        i = 1
        gamma_b=self.walk_geodesic(Input, b)
        while i <= m-1-J and log_density(gamma_b) > level:
            b = b + w
            i = i+1
            gamma_b=self.walk_geodesic(Input, b)
        
        
        return (a,b)
        
        
    def shrinkage(self,a,b,Input, level, log_density):  
        '''
        Shrinkage procedure.
        
        Parameters:
            a,b (float) : Upper and lower bound for the interval to be shrinked.
            log_density (function) : Should take arguments from the space where the shrinkage happens and return real numbers. Levelsets of this function are targeted by the shrinkage procedure.
            walker (function) : Should take float arguments and return points that can be passed to denisty. Used to make the shrinkage steps.
            level (float) : Level belonging to the level set targeted by the shrinkage procedure. Should be between 0 and density(walker(0)).
            
        Returns:
            y : Point on the samping space obtained by the shrinkage procedure
            count: the number of computations before to accept
        '''
        
        
        theta = np.random.uniform(0, b-a)
        theta_h=theta+(a-b)*(theta>b)
        theta_min = theta
        theta_max = theta
        
        y = self.walk_geodesic(Input, theta_h)
        
        count=1
        while log_density(y) <= level:
            
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
            y = self.walk_geodesic(Input, theta_h)

        # not necessary memory, it is to analyze the sampler
        self.counts_shrink[self.nn]=count
        self.chosen_theta[self.nn]=abs(theta)
        return y,count
        
    def sample_velocity(self,x):
        """
        Sample a velocitu on the sphere of R^d
        """
        v=np.random.normal(size=len(x))
        v=v/np.linalg.norm(v)
        return v
    def GSS_Stiefel(self,log_density,X,w, m=1):
        '''
        Transition kernel of the Geodesic slice sampler (GSS) for the Stiefel manifold.
        
        Parameters:
            density (function): Takes dxk-dimensional arrays (= points on the Stiefel manifold) and returns real numbers >= 0. Describes invariant distribution of the GSS.
            X (nxp-dimensional array) : Current point on the Stiefel manifold.
            w (float) : Interval lenght for the stepping-out procedure.
            m (int) : Maximum number of stepping-out steps. Default m = 1.
            
        Returns: 
            (nxp-dimensional array) : Point on the Stiefel manifold obtained by one step of the GSS.
        '''
        
        level = np.log(np.random.uniform(0, 1))+log_density(X) # log_level of acceptance since we compute the log_density
        V= self.sample_velocity(X)
        Input=(X,V)
        a,b = self.stepping_out(w,Input, level=level, log_density=log_density, m=m)
        Y = self.shrinkage(a,b,Input, level=level, log_density=log_density)
        
        return Y
        
    def run_kernel(self,X_0, n_iter,w=1,m=1,log_density=None,use_adapt=False,optimal_rate=0.234):
        '''
        Simulates the trajectory of a Markov chain.
        
        Parameters:
            kernel (function) : Implementation of the transition kernel of the Markov chain.
            X : Initial point.
            n (int) : Lenght of the trajectory.
            
        Returns:
            data (list) : Contains the sampled trajectory.
        '''
        if log_density is None:
            log_density=self.log_prob
        #we initialize some vector to create some statistics on the sampling
        self.counts_shrink=np.zeros(n_iter)
        self.counts_stepping=np.zeros(n_iter)
        self.chosen_theta=np.zeros(n_iter)
        self.nn=0
        #our list of samples is data
        data = []
        prop_w=w
    
        X=X_0
        for i in range(n_iter//20):
            sum=0
            for j in range(20):
                X,count = self.GSS_Stiefel(log_density, X, prop_w,m=m)
                sum=sum+1/count
                self.nn=self.nn+1
                data.append(X)
            rate_X=sum/20
            if use_adapt: # adaptative step to tune w
                adaptive_X = 2*(rate_X > optimal_rate)-1
                prop_w = np.exp(np.log(prop_w) + 0.5*adaptive_X/(1+i)**0.6)
        print("rate_X",rate_X)
        return data


class Sampler_Slice_Sampling_Geodesics():
    

    def __init__(self,log_prob,n,p):
        """
        Parameters:  
            
            log_prob function (nxp-dimensional array-> scalar): log_density on the Stiefel manifold .
            n, p(int) : Dimensionparameters for the Stiefel manifold. It should be n > p.
        
        """
        self.log_prob=log_prob
        self.n=n
        self.p=p
        self.counts_stepping=None
        self.counts_shrink=None
        self.error_reprojection=[]
        self.optimal_rate=0.234 # proposition but not theoric optimal rate
    def null_space(self,X):
        """
        Input :
        X point on the Stiefel n,p manifold 
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
        An element D of the tangent space at X can be expressed as D = XA + X_perp B where A is pxp-dimensional and skew symmetic, B is (n-p)xp dimensional
        and X_perp together with X forms an orthonormal basis (ONB) of R^d.
        
        Parameters:  
            
            X (nxp-dimensional array): Point on the Stiefel manifold at which the tangent sphere is sampled from.
            n, p(int) : Dimensionparameters for the Stiefel manifold. It should be n > p.
        
        Returns:
            V=(A,Q,R) where
            A (pxp-dimensional array)
            Q (nxn -dimensional array) : Orthogonal matrix from the QR-decomposition of X_perp B
            R (nxp-dimensional array) : Upper triangular matrix from the QR-decomposition of X_perp  
            and to_exp (v_l,diag_e,v_r) where
                v_l f(diag_e,t)v_r enable us to compute the geodesic at time t with f a well chosen function
                describe in expm 

        '''
        #Draw uniformly from the sphere
        n=self.n
        p=self.p
        dim_stiefel = int(p*(p-1)/2 + p*(n-p))
        raw_samples = np.random.normal(size = dim_stiefel)
        raw_samples = raw_samples / (np.linalg.norm(raw_samples) + + 1e-100)
    
        
        #Map sample to the tangent space of the Stiefel manifold at X.
        # Use the first p(p-1)/2 samples to make the skew-symmetric matrix A.
        A = np.zeros((p,p))
        ind = np.triu_indices(p, 1)
        A[ind] = raw_samples[:int(p*(p-1)/2)]
        A = A - np.transpose(A)
    

        #Use the remaining samples to make matrix B. Compute QR-decomposition of X_perp B.
        X_perp = self.null_space(X)
        B = np.reshape(raw_samples[int(p*(p-1)/2):], (n-p, p))
        Q,R = qr(X_perp.dot(B))
        V=(A,Q,R)

        # compute quantities to estimate the geodesics starting from X with velocity V
        # we use the antisymetric structure of V
        to_exp=self.pre_walk_geodesic(V)
        
        return V,to_exp

    def proj_V(self,X):
        """Orthogonal projection onto V(n,p) for a matrix X with full rank"""
        u, _, v = np.linalg.svd(X, full_matrices=False)
        return u@v

    def pre_walk_geodesic(self, V):
        '''
        
        Compute some quantities useful to compute geodesics on the stiefel manifold
        we should compute the exponential map of an antisymetric matrices,
        we use the eigen values decompistion for complex matrices
        
        Parameters:

            V ((pxp-dimensional array, dxd-dimensional array, dxk-dimensional array)) : Specifies elemnt of the tangent sphere at X that gives direction of the geodesic.
            Should be of the form (A,Q,R) such that the element of the tangent sphere is given by XA+QR
            where A is skew-symmetric, Q is orthogonal and R an upper-triangular matrix.
        
            
        Returns:
            the svd decomposition of i x arg: v_l,diag_e, v_r' since it is autoadjoint in the complex matrice space
            and v_r is v_r' x vec where vec is a constant vector implied in the geodesics computations
        '''
        n=self.n#d, k (int) : Dimensionparameters for the Stiefel manifold. It should be d > k.
        p=self.p
        # Compute XM + QN, where
        # M = exp[t (A  -R^T)] * Id_k
        # N      [  (R     0)]     0 .
        A,Q,R = V
        arg_top = np.hstack((A, -np.transpose(R)))
        arg_bottom = np.hstack((R, np.zeros((p,p))))
        arg = np.vstack((arg_top, arg_bottom))
        
        vec = np.vstack((np.eye(p), np.zeros((p,p))))
        v_l,diag_e,v_r=self.pre_exp(arg,vec)
        
        return v_l,diag_e,v_r

    def walk_geodesic(self,Input,t):
        X,V,to_exp=Input
        v_l,diag_e,v_r=to_exp
        A,Q,R = V
        n,p=X.shape
        H=self.expm(v_l,diag_e,v_r,t)
        M, N = np.split(H, [p])
        Y = X.dot(M) + Q.dot(N)
         #Reorthogonalise before returning the new point on the Stiefel manifold.
        # We do this to counteract the numerical error induced in each stepp.
        # If we do not do this we will already after two steps "leave" the Stiefel manifold.
    
        Reprojection= self.proj_V(Y)
        # we compute the error of reprojection at each step, we can observe this to inspect instabilities
        self.error_reprojection.append(scl.norm(Reprojection-Y))
        return Reprojection
    def pre_exp(self,M,vec):
        """
        we compute the complex eigenvalues of iM which is autoadjoint as a complex matrix
        and we precompute the product of basis of decomposition with vec
        """
        e,v=np.linalg.eigh(M*ii)
        return v,e,(v.T.conj())@vec
    def expm(self,v_l,e,v_r,t):
        """ 
        this formula combines with pre_exp gives nearly the same result than scipy.expm,
        we use the antisymetric structure to accelerate the computation
        """
        return np.real(v_l@np.diag(np.exp(-t*e*ii))@v_r)
    


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
        
        # Stepping out to the left.
        i = 1
        count=0
        gamma_a=self.walk_geodesic(Input, a)
        
        while i <= J and log_density(gamma_a) > level:
            
            a = a - w
            i = i + 1
            count+=1
            gamma_a=self.walk_geodesic(Input, a)
        
        self.counts_stepping[self.nn]=count 
        
        # Stepping out to the right.
        i = 1
        gamma_b=self.walk_geodesic(Input, b)
        while i <= m-1-J and log_density(gamma_b) > level:
            b = b + w
            i = i+1
            gamma_b=self.walk_geodesic(Input, b)
        
        
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
        
        y = self.walk_geodesic(Input, theta_h)
        
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
            y = self.walk_geodesic(Input, theta_h)

        self.counts_shrink[self.nn]=count
        self.chosen_theta[self.nn]=abs(theta)
        return y,count
        
        
    def GSS_Stiefel(self,log_density,X,w, m=1):
        '''
        Transition kernel of the Geodesic slice sampler (GSS) for the Stiefel manifold.
        
        Parameters:
            log_density (function): Takes nxp-dimensional arrays (= points on the Stiefel manifold) and returns real numbers . Describes invariant distribution of the GSS.
            X (nxp-dimensional array) : Current point on the Stiefel manifold.
            n,p (int) : Dimensionparameters for the Stiefel manifold. It should be n > p.
            w (float) : Interval lenght for the stepping-out procedure.
            m (int) : Maximum number of stepping-out steps. Default m = 1.
            
        Returns: 
            (nxp-dimensional array) : Point on the Stiefel manifold obtained by one step of the GSS.
        '''
        
        level = np.log(np.random.uniform(0, 1))+log_density(X)
        V,to_exp = self.sample_tangent_sphere(X)
        
        Input=(X,V,to_exp)
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
            data (list) : Contains the samples on the Stiefel manifold n,p related to the log_desnity.
        '''
        if log_density is None:
            log_density=self.log_prob
        self.counts_shrink=np.zeros(n_iter)
        self.counts_stepping=np.zeros(n_iter)
        self.chosen_theta=np.zeros(n_iter)
        self.nn=0
        data = [X_0]
        prop_w=w
        #kernel = lambda x : self.GSS_Stiefel(self.p, x, w, self.d,self.k)
        X=X_0
        rate_sum=0
        for i in range(n_iter//20):# we devide by 20 to loop on 20 to estimate the "acceptance ratio"
            
            sum=0
            for j in range(20):
               
                X,count = self.GSS_Stiefel(log_density, X, prop_w,m=m)
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
        return data
    
    
