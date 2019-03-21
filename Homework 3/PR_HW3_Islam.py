
# coding: utf-8

# In[43]:


#Call relevant libraries
import csv
import numpy as np 
import matplotlib.pyplot as plt

#########################################################################################################################
# Function Name: calc_df 
# Function Inputs: var1,size1,var2,size2
# Returns: df
# Description: This function takes in the variances and sizes for two groups of data and calculates the degree of freedom
# using the given formula. The degree of freedom value is then returned via the df variable.
#########################################################################################################################
def calc_df(var1,size1,var2,size2):
    # Calculate the numerator of the formula
    a = (((var1/size1)+(var2/size2))**2)
    # Calculate the denominators of the formula
    b = (((var1/size1)**2)/(size1-1.0))
    c = (((var2/size2)**2)/(size2-1.0))
    # Implement the Degree of Freedom formula
    df = a/(b+c)
    return df

#########################################################################################################################
# Function Name: t_test 
# Function Inputs: mean1,var1,size1,mean2,var2,size2
# Returns: t
# Description: This function takes in the means,variances and sizes for two groups of data and calculates the T score for
# the given data. Since the sign of the score is not important, (the sign depends on the order in which the data is passed)
# the absolute value of the T score is returned as the t variable
#########################################################################################################################
def t_test(mean1,var1,size1,mean2,var2,size2):
    # Implement the T-Score calculation formula
    t = np.abs((mean1-mean2)/(np.sqrt((var1/size1)+(var2/size2))))
    return t

#########################################################################################################################
# Function Name: anova
# Function Inputs: type1,type2,type3,k
# Returns: F
# Description: This function takes in 3 groups of data (as arrays) and computes the Anova score using the given method.
# The relevant means and sizes are calculated within the function. The SSB and SSE values are calculated and the fourth
# input is the K value, which defines df1 i.e. the degree of freedoms. 
#########################################################################################################################
def anova(type1,type2,type3,k):
    # Combined the passed data groups into a combined array
    combined_data=np.array([type1,type2,type3])
    #Calculate the means and sizes of the passed data groups
    type1_mean=np.mean(type1)
    type2_mean=np.mean(type2)
    type3_mean=np.mean(type3)
    type1_size=np.size(type1)
    type2_size=np.size(type2)
    type3_size=np.size(type3)    
    
    # Compute the N value (sample size)
    N=combined_data.size
    #Calculate df1 & df2
    df1=k-1
    df2=N-k
    
    # Compile the means and sizes of the three groups in one array
    combined_mean=np.array([type1_mean,type2_mean,type3_mean])
    combined_sizes=np.array([type1_size,type2_size,type3_size])
    # Compute the overall mean between the groups
    combined_data_mean = np.mean(combined_data)
    # Calculate the SSB using the given formula
    ssb=np.sum(np.multiply(combined_sizes,((combined_mean-combined_data_mean)**2)))
    # Calculate the SSE for each group
    sse_type1=np.sum((type1-type1_mean)**2)
    sse_type2=np.sum((type2-type2_mean)**2)
    sse_type3=np.sum((type3-type3_mean)**2)
    # Sum up the SSE's
    sse=sse_type1+sse_type2+sse_type3
    # Calculate MS1 & MS2 given the SSB & SSE calculated above and df1 & df2
    ms1=ssb/df1
    ms2=sse/df2
    # Finally compute the F-Score
    F=ms1/ms2
    return F


# In[47]:


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

# Extract the Petal Width feature from the groups and store respectively
PW_setosa=np.array([[int(index[0])] for index in setosa])
PW_verginica =np.array([[int(index[0])] for index in verginica])
PW_versicolor=np.array([[int(index[0])] for index in versicolor])

# Extract the Petal Length feature from the groups and store respectively
PL_setosa=np.array([[int(index[1])] for index in setosa])
PL_verginica =np.array([[int(index[1])] for index in verginica])
PL_versicolor=np.array([[int(index[1])] for index in versicolor])

# Extract the Sepal Width feature from the groups and store respectively
SW_setosa=np.array([[int(index[2])] for index in setosa])
SW_verginica =np.array([[int(index[2])] for index in verginica])
SW_versicolor=np.array([[int(index[2])] for index in versicolor])

# Extract the Sepal Width feature from the groups and store respectively
SL_setosa=np.array([[int(index[3])] for index in setosa])
SL_verginica =np.array([[int(index[3])] for index in verginica])
SL_versicolor=np.array([[int(index[3])] for index in versicolor])

#Compute the means for each group and features respectively
PW_setosa_mean = np.mean(PW_setosa)
PL_setosa_mean=np.mean(PL_setosa)
SW_setosa_mean=np.mean(SW_setosa)
SL_setosa_mean=np.mean(SL_setosa)

PW_verginica_mean= np.mean(PW_verginica)
PL_verginica_mean= np.mean(PL_verginica)
SW_verginica_mean= np.mean(SW_verginica)
SL_verginica_mean= np.mean(SL_verginica)

PW_versicolor_mean=np.mean(PW_versicolor)
PL_versicolor_mean=np.mean(PL_versicolor)
SW_versicolor_mean=np.mean(SW_versicolor)
SL_versicolor_mean=np.mean(SL_versicolor)

#Compute the variances for each group and features respectively
PW_setosa_var = np.var(PW_setosa)
PL_setosa_var = np.var(PL_setosa)
SW_setosa_var = np.var(SW_setosa)
SL_setosa_var = np.var(SL_setosa)

PW_verginica_var = np.var(PW_verginica)
PL_verginica_var = np.var(PL_verginica)
SW_verginica_var = np.var(SW_verginica)
SL_verginica_var = np.var(SL_verginica)

PW_versicolor_var=np.var(PW_versicolor)
PL_versicolor_var=np.var(PL_versicolor)
SW_versicolor_var=np.var(SW_versicolor)
SL_versicolor_var=np.var(SL_versicolor)

#Compute the sizes for each group and features respectively
PW_setosa_size = np.size(PW_setosa)
PL_setosa_size = np.size(PL_setosa)
SW_setosa_size = np.size(SW_setosa)
SL_setosa_size = np.size(SL_setosa)

PW_verginica_size = np.size(PW_verginica)
PL_verginica_size = np.size(PL_verginica)
SW_verginica_size = np.size(SW_verginica)
SL_verginica_size = np.size(SL_verginica)

PW_versicolor_size=np.size(PW_versicolor)
PL_versicolor_size=np.size(PL_versicolor)
SW_versicolor_size=np.size(SW_versicolor)
SL_versicolor_size=np.size(SL_versicolor)

# Print all the data separated above (for checking)
print("Means for Petal Width: %f %f %f"%(PW_setosa_mean, PW_verginica_mean,PW_versicolor_mean))
print("Means for Petal Length: %f %f %f"%(PL_setosa_mean, PL_verginica_mean,PL_versicolor_mean))
print("Means for Sepal Width: %f %f %f"%(SW_setosa_mean, SW_verginica_mean,SW_versicolor_mean))
print("Means for Sepal Length: %f %f %f"%(SL_setosa_mean, SL_verginica_mean,SL_versicolor_mean))
print("Variance for Petal Width: %f %f %f"%(PW_setosa_var, PW_verginica_var,PW_versicolor_var))
print("Variance for Petal Length: %f %f %f"%(PL_setosa_var, PL_verginica_var,PL_versicolor_var))
print("Variance for Sepal Width: %f %f %f"%(SW_setosa_var, SW_verginica_var,SW_versicolor_var))
print("Variance for Sepal Length: %f %f %f"%(SL_setosa_var, SL_verginica_var,SL_versicolor_var))
print("Sample size for Petal Width: %f %f %f" %(PW_setosa_size, PW_verginica_size,PW_versicolor_size))
print("Sample size for Petal Width: %f %f %f" %(PL_setosa_size, PL_verginica_size,PL_versicolor_size))
print("Sample size for Sepal Width: %f %f %f" %(SW_setosa_size, SW_verginica_size,SW_versicolor_size))
print("Sample size for Sepal Width: %f %f %f" %(SL_setosa_size, SL_verginica_size,SL_versicolor_size))


# In[49]:


# Empty container to store values
t_scores=[]
df=[]

# Make a container to store the different labels
test_labels=['Setosa & Verginica','Setosa & Versicolor','Verginica & Versicolor']

# Compute the T-Scores for Petal Width feature for the different flower combinations
t_setosa_verginica=t_test(PW_setosa_mean,PW_setosa_var,PW_setosa_size,PW_verginica_mean,PW_verginica_var,PW_verginica_size)
t_setosa_versicolor=t_test(PW_setosa_mean,PW_setosa_var,PW_setosa_size,PW_versicolor_mean,PW_versicolor_var,PW_versicolor_size)
t_verginica_versicolor=t_test(PW_verginica_mean,PW_verginica_var,PW_verginica_size,PW_versicolor_mean,PW_versicolor_var,PW_versicolor_size)
t_scores=np.append(t_scores,t_setosa_verginica)
t_scores=np.append(t_scores,t_setosa_versicolor)
t_scores=np.append(t_scores,t_verginica_versicolor)

# Compute the Degrees of Freedom for Petal Width feature for different flower combinations
df_setosa_verginica = calc_df(PW_setosa_var,PW_setosa_size,PW_verginica_var,PW_verginica_size)
df_setosa_versicolor = calc_df(PW_setosa_var,PW_setosa_size,PW_versicolor_var,PW_versicolor_size)
df_verginica_versicolor = calc_df(PW_verginica_var,PW_verginica_size,PW_versicolor_var,PW_versicolor_size)
df=np.append(df,df_setosa_verginica)
df=np.append(df,df_setosa_versicolor)
df=np.append(df,df_verginica_versicolor)

for i in range(len(test_labels)):
    print("T score for %s: %f" %(test_labels[i],t_scores[i]))
    print("Degress of freedom - df for %s: %F" % (test_labels[i],df[i]))
    
# Using an Alpha of 0.05 and the degrees of freedom calculated above, and looking up T-Distribution Table..
critical_value=2
#Check for Feature Quality
for i in range(len(t_scores)):
    if(t_scores[i]>critical_value):
        print('Petal Width Feature for %s is a good feature'%test_labels[i])
    else:
        print('Petal Width Feature for %s is a bad good feature'%test_labels[i])


# In[53]:


# Overall Container to hold Anova Values
anova_scores=[]
# Make a container to hold the features labels
features = ['Petal Width','Petal Length', 'Sepal Width', 'Sepal Length']
# Compute the Anova scores for the 4 features
anova_PW=anova(PW_setosa,PW_verginica,PW_versicolor,3)
anova_scores=np.append(anova_scores,anova_PW)
anova_PL=anova(PL_setosa,PL_verginica,PL_versicolor,3)
anova_scores=np.append(anova_scores,anova_PL)
anova_SW=anova(SW_setosa,SW_verginica,SW_versicolor,3)
anova_scores=np.append(anova_scores,anova_SW)
anova_SL=anova(SL_setosa,SL_verginica,SL_versicolor,3)
anova_scores=np.append(anova_scores,anova_SL)

#Print the Anova scores
print('Anova score for Petal Width: %f'%anova_PW)
print('Anova score for Petal Length: %f'%anova_PL)
print('Anova score for Sepal Width: %f'%anova_SW)
print('Anova score for Sepal Length: %f'%anova_SL)

# From F-Distribution table, using df1 of 2, df2 of 147 and alpha of 0.05, the critical value for the data set is 2.9957
critical_value=2.9957
for i in range(len(anova_scores)):
    if anova_scores[i]>critical_value:
        print('%s is a good feature'%features[i])
    else:
        print('%s is a bad good feature'%features[i])

