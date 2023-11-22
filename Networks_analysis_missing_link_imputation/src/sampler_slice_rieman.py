import numpy as np
from numba import njit
import src.stiefel as st
import scipy.linalg as scl
from numpy.linalg import qr 
from src import model, model_bin



ii=complex(0,1)
@njit
def sample_tangent_sphere_geo(X):
    '''
    Sample with a gaussian from the tangent space at X.
    An element D of the tangent space at X can be expressed as D = XA + X_perp B where A is kxk-dimensional and skew symmetic, B is (d-k)xk dimensional
    and X_perp together with X forms an orthonormal basis (ONB) of R^d.
    
    Parameters:  
        
        X (dxk-dimensional array): Point on the Stiefel manifold at which the tangent sphere is sampled from.
        n, p(int) : Dimensionparameters for the Stiefel manifold. It should be d > k.
    
    Returns:
        A (pxp-dimensional array)
        Q (nxn -dimensional array) : Orthogonal matrix from the QR-decomposition of X_perp B
        R (nxp-dimensional array) : Upper triangular matrix from the QR-decomposition of X_perp       
    '''
    #Draw uniformly from the sphere
    n,p=X.shape
    dim_stiefel = int(p*(p-1)/2 + p*(n-p))
    
    raw_samples = np.random.standard_normal(dim_stiefel)
    

    #Use the remaining samples to make matrix B. Compute QR-decomposition of X_perp B.
    #X_perp = scl.null_space(np.pad(X, ((0,0), (0, (d-k))))) # can be written in numpy if necessary
    X_perp = null_space(X)
    B = np.reshape(raw_samples[int(p*(p-1)/2):], (n-p, p))
    Q,R = qr(np.ascontiguousarray(X_perp)@np.ascontiguousarray(B)) 
    #Map sample to the tangent space of the Stiefel manifold at X.
    # Use the first k(k-1)/2 samples to make the skew-symmetric matrix A.
    A = np.zeros((p,p))
    ind = np.triu_indices(p, 1)
    for i in range(len(ind[0])):
        A[ind[0][i],ind[1][i]]=raw_samples[i]
    A = A - np.transpose(A)
    
    #we compute quantities related to the exponential
    v_l,e,v_r=pre_walk_geodesic(A,Q,R,p)
    
    return A,Q,R,v_l,e,v_r
@njit
def sample_tangent_sphere(X):
    '''
    Sample uniformly from the tangent sphere at X.
    An element D of the tangent space at X can be expressed as D = XA + X_perp B where A is kxk-dimensional and skew symmetic, B is (d-k)xk dimensional
    and X_perp together with X forms an orthonormal basis (ONB) of R^d.
    
    Parameters:  
        
        X (dxk-dimensional array): Point on the Stiefel manifold at which the tangent sphere is sampled from.
        n, p(int) : Dimensionparameters for the Stiefel manifold. It should be d > k.
    
    Returns:
        A (pxp-dimensional array)
        Q (nxn -dimensional array) : Orthogonal matrix from the QR-decomposition of X_perp B
        R (nxp-dimensional array) : Upper triangular matrix from the QR-decomposition of X_perp       
    '''
    #Draw uniformly from the sphere
    n,p=X.shape
    dim_stiefel = int(p*(p-1)/2 + p*(n-p))
    
    raw_samples = np.random.standard_normal(dim_stiefel)
    #raw_samples = np.random.normal(size = dim_stiefel)
    raw_samples = raw_samples / (np.linalg.norm(raw_samples) + + 1e-100)

    #Use the remaining samples to make matrix B. Compute QR-decomposition of X_perp B.
    #X_perp = scl.null_space(np.pad(X, ((0,0), (0, (d-k))))) # can be written in numpy if necessary
    X_perp = null_space(X)
    B = np.reshape(raw_samples[int(p*(p-1)/2):], (n-p, p))
    Q,R = qr(np.ascontiguousarray(X_perp)@np.ascontiguousarray(B)) 
    #Map sample to the tangent space of the Stiefel manifold at X.
    # Use the first k(k-1)/2 samples to make the skew-symmetric matrix A.
    A = np.zeros((p,p))
    ind = np.triu_indices(p, 1)
    for i in range(len(ind[0])):
        A[ind[0][i],ind[1][i]]=raw_samples[i]
    A = A - np.transpose(A)
    
    #we compute quantities related to the exponential
    v_l,e,v_r=pre_walk_geodesic(A,Q,R,p)
    
    return A,Q,R,v_l,e,v_r
@njit
def null_space(X):
    u, s, vh = np.linalg.svd(X, full_matrices=True)
    #print(u,s,vh)
    n, p = X.shape
    Q = u[:,p:]
        
    return Q
@njit
def pre_walk_geodesic(A,Q,R,p):
    '''
    
    Follow the geodesic through X along direction V on the Stiefel manifold for a given step length.
    
    Parameters:
        X (dxk-dimensional array):  Point on the Stiefel manifold at which the geodesic starts.
        V ((kxk-dimensional array, dxd-dimensional array, dxk-dimensional array)) : Specifies elemnt of the tangent sphere at X that gives direction of the geodesic.
        Should be of the form (A,Q,R) such that the element of the tangent sphere is given by XA+QR
        where A is skew-symmetric, Q is orthogonal and R an upper-triangular matrix.
        t (float) : steplength
        
    Returns:
        Y (dxk-dimensional array) : Result of moving along the geodesic through X in direction V for a steplength of t.
    '''
   
    # Compute XM + QN, where
    # M = exp[t (A  -R^T)] * Id_k
    # N      [  (R     0)]     0 .
    
    arg_top = np.hstack((A, -np.transpose(R)))
    arg_bottom = np.hstack((R, np.zeros((p,p))))
    arg = np.vstack((arg_top, arg_bottom))
    vec = np.vstack((np.eye(p), np.zeros((p,p))))
    v_l,e,v_r=pre_exp(arg,vec)
    return v_l,e,v_r
@njit
def proj_V(X):
    """Orthogonal projection onto V(n,p) for a matrix X with full rank"""
    u, _, v = np.linalg.svd(X, full_matrices=False)
    return u@v
@njit
def pre_exp(M,vec):
    e,v=np.linalg.eigh(M*ii)
    return v,e,(v.T.conj())@vec.astype(np.complex128)
@njit
def expm(v_l,e,v_r,t):
    return np.real(v_l@np.diag(np.exp(-t*e*ii))@v_r)
@njit
def walk_geodesic(X,A,Q,R,v_l,e,v_r,t):
    
    n,p=X.shape
    H=expm(v_l,e,v_r,t)
    M, N = np.split(H, [p])
    Y = np.ascontiguousarray(X).dot(np.ascontiguousarray(M)) + np.ascontiguousarray(Q).dot(np.ascontiguousarray(N))
    #Reorthogonalise before returning the new point on the Stiefel manifold.
    # We do this to counteract the numerical error induced in each stepp.
    # If we do not do this we will already after two steps "leave" the Stiefel manifold.
    Reprojection= proj_V(Y)
    return Reprojection

@njit
def stepping_out(w,X,A,Q,R,v_l,e,v_r,param=None, log_level= None,setting="gaussian", m = 1):
    '''
    Stepping-out procedure.
    
    Parameters:
        density (function) : Should take arguments from the space where the stepping out happens and return real numbers >= 0. Levelsets of this function are targeted by the stepping-out procedure. Is not needed if m=1.
        walker (function) : Should take float arguments and return points that can be passed to denisty. Used to make the stepping out steps. Is not needed if m=1.
        level (float) : Level belonging to the level set targeted by the stepping-out procedure. Should be > 0. Is not needed if m = 1.
        w (float) : Length of the stepping out interval.
        m (int) : Maximum number of stepping out steps. Default 1.
        
    Returns:
        a,b (float) : Determine the interval which results from the stepping out procedure.
    '''
    
    ll_,AA_,theta_=param
    a = - np.random.uniform(0,w)
    b = a + w
    
    if m == 1:
        return (a,b)
        
    J = np.random.randint(0,m)
    
    # Stepping out to the left.
    i = 1
    count=0
    gamma_a=walk_geodesic(X,A,Q,R,v_l,e,v_r, a)
    if setting=="gaussian":
        density=model.log_lk_partial(gamma_a, ll_, AA_, theta_)# à vérifier
    else:
        density=model_bin.log_lk_partial(gamma_a, ll_, AA_, theta_)
    while i <= J and density > log_level:
        a = a - w
        i = i + 1
        count+=1
        gamma_a=walk_geodesic(X,A,Q,R,v_l,e,v_r, a)
        if setting=="gaussian":
            density=model.log_lk_partial(gamma_a, ll_, AA_, theta_)# à vérifier
        else:
            density=model_bin.log_lk_partial(gamma_a, ll_, AA_, theta_)
    
    
    
    # Stepping out to the right.
    i = 1
    gamma_b=walk_geodesic(X,A,Q,R,v_l,e,v_r, b)
    if setting=="gaussian":
        density=model.log_lk_partial(gamma_b, ll_, AA_, theta_)# à vérifier
    else:
        density=model_bin.log_lk_partial(gamma_b, ll_, AA_, theta_)
    while i <= m-1-J and density  > log_level:
        b = b + w
        i = i+1
        gamma_b=walk_geodesic(X,A,Q,R,v_l,e,v_r, b)
        if setting=="gaussian":
            density=model.log_lk_partial(gamma_b, ll_, AA_, theta_)# à vérifier
        else:
            density=model_bin.log_lk_partial(gamma_b, ll_, AA_, theta_)
    
    
    return (a,b)
@njit
def shrinkage(a,b,X,A,Q,R,v_l,e,v_r,param, log_level,setting="gaussian"):  
    '''
    Shrinkage procedure.
    
    Parameters:
        a,b (float) : Upper and lower bound for the interval to be shrinked.
        density (function) : Should take arguments from the space where the shrinkage happens and return real numbers >= 0. Levelsets of this function are targeted by the shrinkage procedure.
        walker (function) : Should take float arguments and return points that can be passed to denisty. Used to make the shrinkage steps.
        level (float) : Level belonging to the level set targeted by the shrinkage procedure. Should be between 0 and density(walker(0)).
        
    Returns:
        y : Point on the samping space obtained by the shrinkage procedure
    '''
    ll_,AA_,theta_=param
    theta = np.random.uniform(0, b-a)
    theta_h=theta+(a-b)*(theta>b)
    theta_min = theta
    theta_max = theta
    
    y = walk_geodesic(X,A,Q,R,v_l,e,v_r, theta_h)

    count=1
    if setting=="gaussian":
        density=model.log_lk_partial(y, ll_, AA_, theta_)# à vérifier
    else:
        density=model_bin.log_lk_partial(y, ll_, AA_, theta_)
    while density <= log_level:
        
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
        y = walk_geodesic(X,A,Q,R,v_l,e,v_r, theta_h)
        if setting=="gaussian":
            density=model.log_lk_partial(y, ll_, AA_, theta_)# à vérifier
        else:
            density=model_bin.log_lk_partial(y, ll_, AA_, theta_)
        
    #self.counts_shrink[self.nn]=count
    #self.chosen_theta[self.nn]=abs(theta)
    return y,count

def GSS_Stiefel(density,X,w, d, k, m=1):
        '''
        Transition kernel of the Geodesic slice sampler (GSS) for the Stiefel manifold.
        
        Parameters:
            density (function): Takes dxk-dimensional arrays (= points on the Stiefel manifold) and returns real numbers >= 0. Describes invariant distribution of the GSS.
            X (dxk-dimensional array) : Current point on the Stiefel manifold.
            d,k (int) : Dimensionparameters for the Stiefel manifold. It should be d > k.
            w (float) : Interval lenght for the stepping-out procedure.
            m (int) : Maximum number of stepping-out steps. Default m = 1.
            
        Returns: 
            (dxk-dimensional array) : Point on the Stiefel manifold obtained by one step of the GSS.
        '''
        
        level = np.random.uniform(0, density(X))
        V = sample_tangent_sphere(X,d,k)
        
        a,b = stepping_out(w,X=X,V=V, level=level, density=density, m=m)
        
        Y = shrinkage(a,b,X=X,V=V, level=level, density=density)
        
        return Y