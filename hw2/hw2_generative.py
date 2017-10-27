import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

#load model
mean_class1 = np.load('mean_class1.npy')
mean_class2 = np.load('mean_class2.npy')
variance = np.load('variance.npy')
class1_num = np.load('class1_num.npy')
class2_num = np.load('class2_num.npy')

#caculate test data
test = open(sys.argv[5],'r',encoding = 'big5')
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



text = open(sys.argv[6], "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()