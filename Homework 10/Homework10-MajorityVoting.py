#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')


# In[58]:


#Open the data file
csv.register_dialect('myDialect',
delimiter = '\t',
skipinitialspace=True)

#Read data and store the data
data_table = []
with open('Fisher.txt', 'r') as csvFile:
    reader = csv.reader(csvFile, dialect='myDialect')
    data_table = list(reader)
csvFile.close()

header = data_table[0]
del data_table[0]

data = np.zeros((len(data_table), len(data_table[0])))

for i in range(0,len(data_table)):
    tmp = data_table[i]
    for j in range(0,len(tmp)):
        data[i,j] = float(tmp[j])

#Seperate the Feature Columns and Label Column        
features=data[:,[1,2,3,4]]
labels=data[:,[0]]


# In[59]:


#Defince Classifiers to be used
d3=tree.DecisionTreeClassifier()
rf=RandomForestClassifier(n_estimators=50, random_state=1)
gp=GaussianProcessClassifier(1.0 * RBF(1.0))
knn=KNeighborsClassifier(n_neighbors=3)
svc=SVC(kernel='linear', C=1)

#Perform Cross validation on Decision Tree Classifier
scores_d3 = cross_val_score(d3, features,np.ravel(labels), cv=10)
accur_crossval_d3=scores_d3.mean()*100
std_crossval_d3=scores_d3.std()*2
print('The Accuracy of the Decision Tree Classifier with 10-fold Cross Validation is : %f'%accur_crossval_d3+'%'+' (+/- %0.2f)'%std_crossval_d3)

#Perform Cross validation on Random Forest Classifier
scores_rf = cross_val_score(rf, features,np.ravel(labels), cv=10)
accur_crossval_rf=scores_rf.mean()*100
std_crossval_rf=scores_rf.std()*2
print('The Accuracy of the Random Forest Classifier with 10-fold Cross Validation is : %f'%accur_crossval_rf+'%'+' (+/- %0.2f)'%std_crossval_rf)

#Perform Cross validation on Gaussian Process  Classifier
scores_gp=cross_val_score(gp, features,np.ravel(labels), cv=10)
accur_crossval_gp=scores_gp.mean()*100
std_crossval_gp=scores_gp.std()*2
print('The Accuracy of the Gaussian Process Classifier with 10-fold Cross Validation is : %f'%accur_crossval_gp+'%'+' (+/- %0.2f)'%std_crossval_gp)

#Perform Cross validation on K-Nearest Neighbour Classifier
scores_knn=cross_val_score(knn, features,np.ravel(labels), cv=10)
accur_crossval_knn=scores_knn.mean()*100
std_crossval_knn=scores_knn.std()*2
print('The Accuracy of the K-Nearest Neighbour Classifier with 10-fold Cross Validation is : %f'%accur_crossval_knn+'%'+' (+/- %0.2f)'%std_crossval_knn)

#Perform Cross validation on Support Vector Machine Classifier
scores_svc=cross_val_score(svc, features,np.ravel(labels), cv=10)
accur_crossval_svc=scores_svc.mean()*100
std_crossval_svc=scores_svc.std()*2
print('The Accuracy of the Support Vector Machine Classifier with 10-fold Cross Validation is : %f'%accur_crossval_svc+'%'+' (+/- %0.2f)'%std_crossval_svc)


# In[63]:


#Perform Majority Voting on the 5 classifiers
mv=VotingClassifier(estimators=[('D3', d3), ('rf', rf), ('gp', gp),('knn',knn),('svc',svc)], voting='hard')
scores_mv = cross_val_score(mv, features,np.ravel(labels), cv=10)
accur_crossval_mv=scores_mv.mean()*100
std_crossval_mv=scores_mv.std()*2
print('The Accuracy of the Majority Vote Classifier with 10-fold Cross Validation is : %f'%accur_crossval_mv+'%'+' (+/- %0.2f)'%std_crossval_mv)


# In[ ]:




