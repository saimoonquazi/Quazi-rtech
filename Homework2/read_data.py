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
		data[i,j] = int(tmp[j])
		    
type0 = data[data[:,0] == 0,:]
type1 = data[data[:,0] == 1,:]
type2 = data[data[:,0] == 2,:]

print(type0)