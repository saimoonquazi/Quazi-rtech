#!/usr/bin/env python
# coding: utf-8

# Reading in the data and seperating the features and the labels only. No splitting is required since cross validation takes care of that. 

# In[26]:


import csv
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np


# In[27]:


# Load the Fisher data file
csv.register_dialect('myDialect',
delimiter = '\t',
skipinitialspace=True)

# Read data
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

# Seperate Features and Labels        
feature=data[:,[1,2,3,4]]
labels=data[:,[0]]


# In[28]:


# Define the Leave Out One Object
LOO=LeaveOneOut()
# Compute the number of iterations for the leave one out validation (same as sample size)
number_of_iterations=LOO.get_n_splits(feature)
#Define the Decision Tree Classifier
d3=tree.DecisionTreeClassifier()


# In[29]:


#Define a total score variable
total_score=0;
# Perform Leave one out validation on the Decision Tree Classifier
for train_index,test_index in LOO.split(feature):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_features, test_features = feature[train_index], feature[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]
   
    clf=d3.fit(train_features,train_labels)
    total_score+=clf.score(test_features,test_labels)

#Compute the mean accuracy
mean_score=(total_score/number_of_iterations)*100
print('The Accuracy of the Decision Tree Classifier with Leave One Out Validation is : %f' %mean_score+'%')


# In[30]:


# Perform 10-fold Cross Validation with the Decision Tree Classifier
scores_d3 = cross_val_score(d3, feature, labels, cv=10)
accur_crossval_d3=scores_d3.mean()*100
std_crossval_d3=scores_d3.std() * 2
print('The Accuracy of the Decision Tree Classifier with 10-fold Cross Validation is : %f'%accur_crossval_d3+'%'+' (+/- %0.2f)'%std_crossval_d3)


# Perform Leave One Out validation on PCA-Decision Tree classifier. Does not work so well in my opinion. You will get lots of warnings so used "np.seterr" to ignore the warnings. 

# In[35]:


np.seterr(divide='ignore', invalid='ignore')

total_score=0
pca=PCA(n_components=1)
for train_index,test_index in LOO.split(feature):
    train_features, test_features = feature[train_index], feature[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]
    
    sc=StandardScaler()
    train_set=sc.fit_transform(train_features)
    test_set=sc.fit_transform(test_features)

    pca_train_set= pca.fit_transform(train_set) 
    pca_test_set=pca.fit_transform(test_set)
    
    clf_pca=d3.fit(pca_train_set,train_labels)
    prediction_pca=clf_pca.predict(pca_test_set)
    total_score+=accuracy_score(test_labels,prediction_pca) 

mean_score=(total_score/number_of_iterations)*100
print('The Accuracy of the Decision Tree Classifier + PCA with Leave One Out Validation is : %f' %mean_score+'%')


# Cross Validation with 10 folds on the PCA-Decision Tree Classifier

# In[32]:


feature_set=sc.fit_transform(feature)

pca_features= pca.fit_transform(feature_set) 

scores_d3PCA = cross_val_score(d3, pca_features, labels, cv=10)
accur_crossval_d3PCA=scores_d3PCA.mean()*100
std_crossval_d3PCA=scores_d3PCA.std() * 2
print('The Accuracy of the Decision Tree Classifier + PCA with 10-fold Cross Validation is : %f'%accur_crossval_d3PCA+'%'+' (+/- %0.2f)'%std_crossval_d3PCA)


# Perform Leave One Out validation for the LDA - Decision Tree Classifier

# In[33]:


lda=LDA(n_components=2)

total_score=0
for train_index,test_index in LOO.split(feature):
    train_features, test_features = feature[train_index], feature[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]
    
    lda_train_set=lda.fit_transform(train_features,np.ravel(train_labels))
    lda_test_set = lda.transform(test_features)
    
    clf_lda=d3.fit(lda_train_set,train_labels)
    prediction_lda=clf_lda.predict(lda_test_set)
    total_score+=accuracy_score(test_labels,prediction_lda) 
#    total_score+=clf_pca.score(pca_test_set,test_labels)

mean_score=(total_score/number_of_iterations)*100
print('The Accuracy of the Decision Tree Classifier + LDA with Leave One Out Validation is : %f' %mean_score+'%')


# Perform Cross Validation for 10 folds for the LDA-Decision Tree Classifier

# In[34]:


lda_features=lda.fit_transform(feature,np.ravel(labels)) 

scores_d3LDA = cross_val_score(d3, lda_features, labels, cv=10)
accur_crossval_d3LDA=scores_d3LDA.mean()*100
std_crossval_d3LDA=scores_d3LDA.std() * 2
print('The Accuracy of the Decision Tree Classifier with 10-fold Cross Validation is : %f'%accur_crossval_d3LDA+'%'+' (+/- %0.2f)'%std_crossval_d3LDA)


# In[ ]:




