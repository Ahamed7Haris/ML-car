import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score
import sklearn.metrics as smk
a=pd.read_csv('Spots.csv')
a['valence']=a['valence'].apply(lambda x: 0 if x<0.5 else( 1 if 0.4<=x <0.7 else 1))
print(a['valence'])
a.to_csv('Spots.csv')
ad=pd.read_csv('Spots.csv')
print(a.head(),a.shape,'\n',a.describe)
sns.FacetGrid(a,hue='mode',height=6).map(plt.scatter,'tempo','valence').add_legend()
a.to_csv('Irris.csv',index=False)
ad=pd.read_csv('Irris.csv')
x=ad[['acousticness']].values
y=ad[['valence']].values
y=y.ravel()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y,'\t',y_pred)
print(y.shape,'\t',y_pred.shape)
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)
conf=confusion_matrix(y_test,y_pred)
print(conf)
print(report)
print("Accuracy:", accuracy)
print("f1 score:",f1_score(y_test,y_pred,average='binary'))
plt.show()



