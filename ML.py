import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
d=pd.read_csv("Device storage/ml/CarPrice_Assignment.csv")
print(d.dtypes)
X=d['peakrpm'].values.reshape(-1,1)
Y=d['price'].values.reshape(-1,1)
Z=d['horsepower'].values.reshape(-1,1)
ad=KMeans(n_clusters=5,random_state=42)
ad.fit(X)
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
cen=ad.cluster_centers_
cf=ad.labels_
ax.scatter(X,Y,Z,c=cf,cmap='viridis')
ax.set_xlabel('peak rpm')
ax.set_zlabel('horse power')
ax.set_ylabel('price')
plt.show()
ax=4500
az=130
pre={X,Y,Z}
print(pre)
cs=StandardScaler()
sd=cs.fit_transform(pre)
ip=cs.transform([[ax,0,az]])
ic=a.predict(ip)
ccl=a.cluster_centers_[ip]
av=cluster_center[0,2]
print(av)
