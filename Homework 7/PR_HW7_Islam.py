
# coding: utf-8

# In[7]:


import csv
import numpy as np 
import matplotlib.pyplot as plt

csv.register_dialect('myDialect',
delimiter = '\t',
skipinitialspace=True)

# read data
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

# MAKE TRAIN AND TEST DATA
# spilt class
setosa = data[data[:,0] == 0,:]
verginica = data[data[:,0] == 1,:]
versicolor = data[data[:,0] == 2,:]

data_split = 0.8

# get index for splitting the data
samples_setosa = int(setosa.shape[0]*data_split)
samples_verginica = int(verginica.shape[0]*data_split)
samples_versicolor = int(versicolor.shape[0]*data_split)

# make train dataset
train_setosa = setosa[0:samples_setosa,:]
train_verginica = verginica[0:samples_verginica,:]
train_versicolor = versicolor[0:samples_versicolor,:]

train_data = np.concatenate((train_setosa, train_verginica, train_versicolor), axis=0)

# make test dataset
test_setosa = setosa[samples_setosa:,:]
test_verginica = verginica[samples_verginica:,:]
test_versicolor = versicolor[samples_versicolor:,:]

test_data = np.concatenate((test_setosa, test_verginica, test_versicolor), axis=0)


setosa = train_data[train_data[:,0] == 0,1:5]
verginica = train_data[train_data[:,0] == 1,1:5]
versicolor = train_data[train_data[:,0] == 2,1:5]

plt.plot(setosa[:,0], setosa[:,1], 'ro', verginica[:,0], verginica[:,1], 'bs', versicolor[:,0], versicolor[:,1], 'g^')
plt.xlabel('F1')
plt.ylabel('F2')
plt.title('Original Data')
plt.show()


# In[8]:


# create a 2D array
setosa_mean = np.array([np.mean(setosa, axis=0)])
verginica_mean = np.array([np.mean(verginica, axis=0)])
versicolor_mean = np.array([np.mean(versicolor, axis=0)])

setosa_mean = setosa_mean.transpose()
verginica_mean = verginica_mean.transpose()
versicolor_mean = versicolor_mean.transpose()

total_mean = (setosa_mean+verginica_mean+versicolor_mean)/3.0

S1 = np.cov((setosa[:,0], setosa[:,1], setosa[:,2], setosa[:,3]))
S2 = np.cov((verginica[:,0], verginica[:,1], verginica[:,2], verginica[:,3]))
S3 = np.cov((versicolor[:,0], versicolor[:,1], versicolor[:,2], versicolor[:,3]))
# Within class scatter matrix
Sw = S1 + S2 + S3

r,N1 = setosa.shape
r,N2 = verginica.shape
r,N3 = versicolor.shape
m1 = setosa_mean-total_mean
m2 = verginica_mean-total_mean
m3 = versicolor_mean-total_mean
Sb1 = N1*np.matmul(m1,m1.transpose())
Sb2 = N2*np.matmul(m2,m2.transpose())
Sb3 = N3*np.matmul(m3,m3.transpose())

# Between-class scatter matrix
Sb = Sb1 + Sb2 + Sb3;

# LDA projection
SwSb = np.matmul(np.linalg.pinv(Sw),Sb)

# Projection vector
w, v = np.linalg.eig(SwSb)

# sort according to eigenvalues
index = np.argsort(-w)

# Use all eigenvectors for full reconstruction
W1 = v[:,index]

# Reduce dimension
W1[:,3] = 0

# Project data samples along the projection axes
new_setosa = np.matmul(setosa,W1)
new_verginica = np.matmul(verginica,W1)
new_versicolor = np.matmul(versicolor,W1)

plt.plot(new_setosa[:,0], new_setosa[:,1], 'ro', new_verginica[:,0], new_verginica[:,1], 'bs', new_versicolor[:,0], new_versicolor[:,1], 'g^')
plt.xlabel('F1')
plt.ylabel('F2')
plt.title('Transformed Data')
plt.show()


# In[9]:


# Project TEST DATA
test_project = np.matmul(test_data[:,1:5],W1)

# Compute the mean of the prjected data
mean_setosa = np.mean(new_setosa, axis = 0)
mean_verginica = np.mean(new_verginica, axis = 0)
mean_versicolor = np.mean(new_versicolor, axis = 0)

acu = 0
acu2 = 0 

for i in range(0, test_data.shape[0]):
    s = test_project[i,:] #smaple
    
    #Euclidean distance
    dist1 = np.sqrt(((s[0]-mean_setosa[0])**2) + ((s[1]-mean_setosa[1])**2) + ((s[2]-mean_setosa[2])**2))
    dist2 = np.sqrt(((s[0]-mean_verginica[0])**2) + ((s[1]-mean_verginica[1])**2) + ((s[2]-mean_verginica[2])**2))
    dist3 = np.sqrt(((s[0]-mean_versicolor[0])**2) + ((s[1]-mean_versicolor[1])**2) + ((s[2]-mean_versicolor[2])**2))

    d = np.array([dist1, dist2, dist3])
    idx = np.argsort(d)
    predicted_class = idx[0]
    
    if(test_data[i,0] == predicted_class):
        acu = acu+1
    
    #Manhattan distance
    dist4 = np.absolute(s[0]-mean_setosa[0]) + np.absolute(s[1]-mean_setosa[1]) + np.absolute(s[2]-mean_setosa[2])
    dist5 = np.absolute(s[0]-mean_verginica[0]) + np.absolute(s[1]-mean_verginica[1]) + np.absolute(s[2]-mean_verginica[2])
    dist6 = np.absolute(s[0]-mean_versicolor[0]) + np.absolute(s[1]-mean_versicolor[1]) + np.absolute(s[2]-mean_versicolor[2])

    d2 = np.array([dist4, dist5, dist6])
    idverginica = np.argsort(d2)
    predicted_class2 = idverginica[0]
    
    if(test_data[i,0] == predicted_class2):
        acu2 = acu2+1   
print("Accuracy for euclidean distance: %f" %(acu/float(test_project.shape[0])))
print("Accuracy for Manhattan distance: %f" %(acu2/float(test_project.shape[0])))

