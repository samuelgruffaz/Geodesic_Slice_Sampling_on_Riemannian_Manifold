import numpy as np
import matplotlib.pyplot as plt

from Sampler_class import *
from Sampler_grassman import *




n,p=3,2

log_prob = lambda x : 0 

#we define the class of the slice sampling sampler
sampler=Sampler_Slice_Sampling_Geodesics(log_prob,n,p)
TT=30
# STIEFEL GEODESICS
data=[]
X_0 = np.array([[1.0,0],[0,0],[0,1.0]])
V,to_exp= sampler.sample_tangent_sphere(X_0)
(A,Q,R)=V
    
(vv,e,dd)=to_exp
    
Input=(X_0, V,to_exp)
walker = lambda t : sampler.walk_geodesic(Input, t)
    
for j in range(TT):
    data.append(walker(2*j*np.pi/TT))
    
    
datax =[]
datay = []
dataz = []
for point in data:
    datax.append(point[0,0])
    datay.append(point[1,0])
    dataz.append(point[2,0])
    datax.append(point[0,1])
    datay.append(point[1,1])
    dataz.append(point[2,1])

fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.scatter(datax,datay,dataz,c=np.arange(len(datax)), cmap = 'viridis',alpha=0.8)
ax.scatter(1,0,0,"*", color = "green",label='1,0,0',alpha=1,s=100)
ax.scatter(0,0,1,"*", color = "red",label='0,0,1',alpha=1,s=100)
plt.legend()
plt.show()



# GRASSMAN GEODESICS

data=[]
sampler=Sampler_Slice_Sampling_Grassman(log_prob,n,p)



(X_V,U,sigma,VT)= sampler.sample_tangent_sphere(X_0)
Input=(X_V,U,sigma,VT)
walker = lambda t : sampler.walk_geodesic(Input, t)
for i in range(TT):
    data.append(walker(2*np.pi*i/TT))

datax =[]
datay = []
dataz = []
for point in data:
    datax.append(point[0,0])
    datay.append(point[1,0])
    dataz.append(point[2,0])
    datax.append(point[0,1])
    datay.append(point[1,1])
    dataz.append(point[2,1])

fig = plt.figure(2)
ax = fig.add_subplot(projection='3d')
ax.scatter(datax,datay,dataz,c=np.arange(len(datax)), cmap = 'viridis',alpha=0.8)
ax.scatter(1,0,0,"*", color = "green",label='1,0,0',alpha=1,s=100)
ax.scatter(0,0,1,"*", color = "red",label='0,0,1',alpha=1,s=100)
plt.legend()
plt.show()