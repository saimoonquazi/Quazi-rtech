
# coding: utf-8

# In[10]:


#Import relevant libraries
import csv
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Open the textfile and read the lines into a container
with open("Fisher.txt") as f:
    content = f.readlines()

#Seperate the first row that contains the titles only    
titles = content[0].split()
#Seperate the rest of the data into a list 
lines = [line.split() for line in content[1:]]

#Seperate all the rows that correspond to the setosa flower label
setosa = np.array([[int(flower[1]), int(flower[2]), int(flower[3]), int(flower[4])] for flower in lines if flower[0] is '0'])
#Seperate all the rows that correspond to the verginica flower label
verginica = np.array([[int(flower[1]), int(flower[2]), int(flower[3]), int(flower[4])] for flower in lines if flower[0] is '1'])
#Seperate all the rows that correspond to the vericolor flower label
versicolor = np.array([[int(flower[1]), int(flower[2]), int(flower[3]), int(flower[4])] for flower in lines if flower[0] is '2'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(setosa[:,0], setosa[:,1], setosa[:,2],setosa[:,3], c='r', marker='o')
ax.scatter(verginica[:,0], verginica[:,1], verginica[:,2],verginica[:,3], c='b', marker='^')
ax.scatter(versicolor[:,0], versicolor[:,1], versicolor[:,2],versicolor[:,3], c='g', marker='*')

ax.set_xlabel('Features of Setosa')
ax.set_ylabel('Features of Verginica')
ax.set_zlabel('Features of Versicolor')

plt.show()
# create a 2D array
setosa_mean = np.array([np.mean(setosa, axis=0)])
verginica_mean = np.array([np.mean(verginica, axis=0)])
versicolor_mean = np.array([np.mean(versicolor, axis=0)])

setosa_mean = setosa_mean.transpose()
verginica_mean = verginica_mean.transpose()
versicolor_mean = versicolor_mean.transpose()

total_mean = (setosa_mean+verginica_mean+versicolor_mean)/3.0

S1 = np.cov((setosa[:,0], setosa[:,1],setosa[:,2],setosa[:,3]))
S2 = np.cov((verginica[:,0], verginica[:,1],verginica[:,2],verginica[:,3]))
S3 = np.cov((versicolor[:,0], versicolor[:,1],versicolor[:,2],versicolor[:,3]))
# within class scatter matrix
Sw = S1 + S2 + S3


r,N1 = X1.shape
r,N2 = X2.shape
r,N3 = X3.shape
m1 = setosa_mean-total_mean
m2 = verginica_mean-total_mean
m3 = versicolor_mean-total_mean
Sb1 = N1*np.matmul(m1,m1.transpose())
Sb2 = N2*np.matmul(m2,m2.transpose())
Sb3 = N3*np.matmul(m3,m3.transpose())
# between-class scatter matrix
Sb = Sb1 + Sb2 + Sb3;

# LDA projection
SwSb = np.matmul(np.linalg.pinv(Sw),Sb)

# projection vector
w, v = np.linalg.eig(SwSb)
# w - eigenvalues
# v - eigenvectors
print("Eigenvalues")
print(w)
print("Eigenvectors")
print(v)
# sort according to eigenvalues
index = np.argsort(-w)
print("Eigenvalues idx")
print(index)
# use all eigenvectors for full reconstruction
W1 = v[:,index]
print("Weight vector")
print(W1)

# project data samplesalong he projection axes
new_setosa = np.matmul(setosa,W1)
new_verginica = np.matmul(verginica,W1)
new_versicolor = np.matmul(versicolor,W1)



# In[12]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(new_setosa[:,0], new_setosa[:,1], new_setosa[:,2], c='r', marker='o')
ax.scatter(new_verginica[:,0], new_verginica[:,1], new_verginica[:,2], c='b', marker='^')
ax.scatter(new_versicolor[:,0], new_versicolor[:,1], new_versicolor[:,2], c='g', marker='*')

ax.set_xlabel('PW,PL,SW for Setosa')
ax.set_ylabel('PW,PL,SW for Verginica')
ax.set_zlabel('PW,PL,SW for Versicolor')

plt.show()

