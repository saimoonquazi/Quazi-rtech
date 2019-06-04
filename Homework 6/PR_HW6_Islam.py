
# coding: utf-8

# In[3]:


import csv
import numpy as np 

csv.register_dialect('myDialect',delimiter = '\t',skipinitialspace=True)
# Read data
data_table = []
# Opening File Data from Source
with open('Fisher.txt', 'r') as csvFile:
    reader = csv.reader(csvFile, dialect='myDialect')
    data_table = list(reader)
csvFile.close()

header = data_table[0]
del data_table[0]
data = np.zeros((len(data_table), len(data_table[0])))

# Reading Data from file
for i in range(0,len(data_table)):
    tmp = data_table[i]
    for j in range(0,len(tmp)):
        data[i,j] = int(tmp[j])
        
# Split class
setosa = data[data[:,0] == 0]
verginica = data[data[:,0] == 1]
versicolor = data[data[:,0] == 2]

# Get index for splitting data 80% - 20%
data_split = 0.8
samples_setosa = int(round(setosa.shape[0]*data_split))
samples_verginica = int(round(verginica.shape[0]*data_split))
samples_versicolor = int(round(versicolor.shape[0]*data_split))

# Seperate training dataset
train_setosa = setosa[0:samples_setosa,:]   
train_verginica = verginica[0:samples_verginica,:]
train_versicolor = versicolor[0:samples_versicolor,:] 

train_data = np.concatenate((train_setosa,train_verginica, train_versicolor), axis=0)

# Seperate test dataset
test_setosa = setosa[samples_setosa:,:]
test_verginica = verginica[samples_verginica:,:]
test_versicolor = versicolor[samples_versicolor:,:]

test_data = np.concatenate((test_setosa,test_verginica, test_versicolor), axis=0)


# In[4]:



# Compute the mean of all the classes
setosa_mean= np.mean(train_setosa[:,1:], axis=0)
verginica_mean= np.mean(train_verginica[:,1:], axis=0)
versicolor_mean= np.mean(train_versicolor[:,1:], axis=0)

acu = 0
acu2 = 0

for i in range(0, test_data.shape[0]):
    #Take Current Sample
    s = test_data[i,1:] 
    
    #Compute Euclidean distances
    dist1 = np.sqrt(((s[0]-setosa_mean[0])**2) + ((s[1]-setosa_mean[1])**2) + ((s[2]-setosa_mean[2])**2) + ((s[3]-setosa_mean[3])**2))
    dist2 = np.sqrt(((s[0]-verginica_mean[0])**2) + ((s[1]-verginica_mean[1])**2) + ((s[2]-verginica_mean[2])**2) + ((s[3]-verginica_mean[3])**2))
    dist3 = np.sqrt(((s[0]-versicolor_mean[0])**2) + ((s[1]-versicolor_mean[1])**2) + ((s[2]-versicolor_mean[2])**2) + ((s[3]-versicolor_mean[3])**2))

    # Temporarily store distances in an array and sort find the indices that sort the array
    d = np.array([dist1, dist2, dist3])
    idx = np.argsort(d)
    # The smallest Euclidean distance is the prediction, in this case dist1 refers to setosa, and so on...
    predicted_class = idx[0]

    if(test_data[i,0] == predicted_class):
        acu = acu+1
    # Compute Manhattan distance
    dist4 = np.absolute(s[0]-setosa_mean[0]) + np.absolute(s[1]-setosa_mean[1]) + np.absolute(s[2]-setosa_mean[2]) + np.absolute(s[3]-setosa_mean[3])
    dist5 = np.absolute(s[0]-verginica_mean[0]) + np.absolute(s[1]-verginica_mean[1]) + np.absolute(s[2]-verginica_mean[2]) + np.absolute(s[3]-verginica_mean[3])
    dist6 = np.absolute(s[0]-versicolor_mean[0]) + np.absolute(s[1]-versicolor_mean[1]) + np.absolute(s[2]-versicolor_mean[2]) + np.absolute(s[3]-versicolor_mean[3])
    
    # Temporarily store distances in an array and sort find the indices that sort the array
    d2 = np.array([dist4, dist5, dist6])
    idx2 = np.argsort(d2)
    # The smallest Euclidean distance is the prediction, in this case dist1 refers to setosa, and so on...
    predicted_class2 = idx2[0]
    
    if(test_data[i,0] == predicted_class2):
        acu2 = acu2+1   
                
print("Task 1 - Minimum Distance Classifier (Baseline):")
print("Accuracy for Euclidean distance: %f" %(acu/float(test_data.shape[0])))
print("Accuracy for Manhattan distance: %f" %(acu2/float(test_data.shape[0])))


# In[5]:


# Shuffle the data randomly
np.random.seed(1222)
np.random.shuffle(data)
#Split the dataset 80% and 20%
data_split = 0.8
samples_training = int(round(data.shape[0]*data_split))
train_data = data[0:samples_training,:]
test_data = data[samples_training:,:]

# Save feature data as a vector for training data
pw_train = train_data[:,1]
pl_train = train_data[:,2]
sw_train = train_data[:,3]
sl_train = train_data[:,4]

# Compute the mean of the training feature data
pw_train_mean = np.mean(pw_train)
pl_train_mean = np.mean(pl_train)
sw_train_mean = np.mean(sw_train)
sl_train_mean = np.mean(sl_train)

# Shift the data 
pw_train_shifted = pw_train-pw_train_mean
pl_train_shifted = pl_train-pl_train_mean
sw_train_shifted = sw_train-sw_train_mean
sl_train_shifted = sl_train-sl_train_mean

# Save feature data as a vector for test data
pw_test = test_data[:,1]
pl_test = test_data[:,2]
sw_test = test_data[:,3]
sl_test = test_data[:,4]

# Compute the mean of the test feature data
pw_test_mean = np.mean(pw_test)
pl_test_mean = np.mean(pl_test)
sw_test_mean = np.mean(sw_test)
sl_test_mean = np.mean(sl_test)

# Shift the data
pw_test_shifted = pw_test-pw_test_mean
pl_test_shifted = pl_test-pl_test_mean
sw_test_shifted = sw_test-sw_test_mean
sl_test_shifted = sl_test-sl_test_mean

# Compute the covariance matrix
cov_mat = np.cov((pw_train_shifted,pl_train_shifted,sw_train_shifted,sl_train_shifted))

w, v = np.linalg.eig(cov_mat)

# Sort according to eigenvalues
index = np.argsort(-w)

# Use all eigenvectors for full reconstruction
feature_vector = v[:,index]

RowFeatureVector = np.transpose(feature_vector)
RowZeroMeanData = np.array([pw_train_shifted, pl_train_shifted, sw_train_shifted, sl_train_shifted])
RowZeroMeanData_test = np.array([pw_test_shifted, pl_test_shifted, sw_test_shifted, sl_test_shifted])

FinalData = np.transpose(np.matmul(RowFeatureVector, RowZeroMeanData))
FinalData_test = np.transpose(np.matmul(RowFeatureVector, RowZeroMeanData_test))

#Split class
# this is for training data
final_setosa = FinalData[train_data[:,0] == 0,:]
final_verginica = FinalData[train_data[:,0] == 1,:]
final_versicolor = FinalData[train_data[:,0] == 2,:]

#get class mean value
final_setosa_mean= np.mean(final_setosa, axis=0)
final_verginica_mean= np.mean(final_verginica, axis=0)
final_versicolor_mean= np.mean(final_versicolor, axis=0)


final_acu = 0
final_acu2 = 0

for i in range(0, FinalData_test.shape[0]):
    #L2 norm
    sample = FinalData_test[i,:] #smaple
    #Euclidean distance
    distance1 = np.sqrt(((sample[0]-final_setosa_mean[0])**2) + ((sample[1]-final_setosa_mean[1])**2) + ((sample[2]-final_setosa_mean[2])**2) + ((sample[3]-final_setosa_mean[3])**2))
    distance2 = np.sqrt(((sample[0]-final_verginica_mean[0])**2) + ((sample[1]-final_verginica_mean[1])**2) + ((sample[2]-final_verginica_mean[2])**2) + ((sample[3]-final_verginica_mean[3])**2))
    distance3 = np.sqrt(((sample[0]-final_versicolor_mean[0])**2) + ((sample[1]-final_versicolor_mean[1])**2) + ((sample[2]-final_versicolor_mean[2])**2) + ((sample[3]-final_versicolor_mean[3])**2))

    final_distance = np.array([distance1, distance2, distance3])
    final_idx = np.argsort(final_distance)
    final_predicted_class = final_idx[0]

    if(test_data[i,0] == final_predicted_class):
        final_acu = final_acu+1

    #Manhattan distance
    distance4 = np.absolute(sample[0]-final_setosa_mean[0]) + np.absolute(sample[1]-final_setosa_mean[1]) + np.absolute(sample[2]-final_setosa_mean[2]) + np.absolute(sample[3]-final_setosa_mean[3])
    distance5 = np.absolute(sample[0]-final_verginica_mean[0]) + np.absolute(sample[1]-final_verginica_mean[1]) + np.absolute(sample[2]-final_verginica_mean[2]) + np.absolute(sample[3]-final_verginica_mean[3])
    distance6 = np.absolute(sample[0]-final_versicolor_mean[0]) + np.absolute(sample[1]-final_versicolor_mean[1]) + np.absolute(sample[2]-final_versicolor_mean[2]) + np.absolute(sample[3]-final_versicolor_mean[3])
    
    final_distance2 = np.array([distance4, distance5, distance6])
    final_idx2 = np.argsort(final_distance2)
    final_predicted_class2 = final_idx2[0]
    
    if(test_data[i,0] == final_predicted_class2):
        final_acu2 = final_acu2+1   

print("Task 2: PCA and Minimum Distance Classifier:")
print("Accuracy for Euclidean distance: %f" %(final_acu/float(test_data.shape[0])))
print("Accuracy for Manhattan distance: %f" %(final_acu2/float(test_data.shape[0])))

