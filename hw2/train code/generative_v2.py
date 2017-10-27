import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

train_x = open('data/X_train','r', encoding='big5')
train_y = open('data/Y_train','r', encoding='big5')

row_x = csv.reader(train_x)
next(row_x,None) #skip the first line
row_y = csv.reader(train_y)
next(row_y,None) #skip the first line

#get out label and feature data
feature = [];
label = [];

for r in row_y:
	label.append(float(r[0]))


now_row = 0
for r in row_x:
	feature.append([])
	for i in range(len(r)):
		feature[now_row].append(float(r[i]))
	now_row = now_row + 1

feature = np.array(feature)
label = np.array(label)

train_x.close()
train_y.close()



#seperate data of different class
#label 0 stands for < 50k
train_class1 = [] # < 50 k
train_class2 = [] # >= 50k
index_class1 = 0
index_class2 = 0
for i in range(len(label)):
	if(label[i] == 0):
		train_class1.append([])
		train_class1[index_class1] = feature[i]
		index_class1 = index_class1 + 1
	elif(label[i] == 1):
		train_class2.append([])
		train_class2[index_class2] = feature[i]
		index_class2 = index_class2 + 1

#caculate the gussian distribution model for class1 and class2
#mean is 1*106dim array
mean_class1 = []
mean_class2 = []
variance_class1 = np.zeros((len(train_class1[0]),len(train_class1[0])))
variance_class2 = np.zeros((len(train_class2[0]),len(train_class2[0])))
variance = []

for i in np.transpose(train_class1):
	mean_class1.append(np.mean(i))
for i in np.transpose(train_class2):
	mean_class2.append(np.mean(i))

for i in range(len(train_class1)):
	variance_class1 = variance_class1 + np.dot(np.transpose([train_class1[i] - mean_class1]) ,[(train_class1[i] - mean_class1)])
variance_class1 = variance_class1 / len(train_class1)
for i in range(len(train_class2)):
	variance_class2 = variance_class2 + np.dot(np.transpose([train_class2[i] - mean_class2]) ,[(train_class2[i] - mean_class2)])
variance_class2 = variance_class2 / len(train_class2)

total_data = len(train_class1) + len(train_class2)
class1_num = len(train_class1)
class2_num = len(train_class2)
variance = float(class1_num) * variance_class1 / total_data + float(class2_num) * variance_class2 / total_data

mean_class1 = np.array(mean_class1)
mean_class2 = np.array(mean_class2)
variance = np.array(variance)
data_num = len(train_class1[0])

#save model 
np.save('mean_class1.npy',mean_class1)
np.save('mean_class2.npy',mean_class2)
np.save('variance.npy',variance)
np.save('class1_num.npy',class1_num)
np.save('class2_num.npy',class2_num)

mean_class1 = np.load('mean_class1.npy')
mean_class2 = np.load('mean_class2.npy')
variance = np.load('variance.npy')
class1_num = np.load('class1_num.npy')
class2_num = np.load('class2_num.npy')

#caculate test data
test = open('data/X_test','r',encoding = 'big5')
row = csv.reader(test)
next(row,None)

test_x = []

index = 0
for r in row:
	test_x.append([])
	for i in range(len(r)):
		test_x[index].append(float(r[i]))
	index = index + 1
test_x = np.array(test_x)


ans = []
id = 1

sigma_inverse = np.linalg.inv(variance)
w = np.dot((mean_class1 - mean_class2) , sigma_inverse)
x = test_x.T
b = (-0.5) * np.dot(np.dot([mean_class1] , sigma_inverse) , mean_class1) +\
	 (0.5) * np.dot(np.dot([mean_class2] , sigma_inverse) , mean_class2) + np.log(float(class1_num)/class2_num)
a = np.dot(w,x) + b
y = 1 - np.around(sigmoid(a))


for i,value in enumerate(y):
	ans.append([])
	ans[i].append(i+1)
	ans[i].append(int(value))



filename = "result/generative_normalization.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

