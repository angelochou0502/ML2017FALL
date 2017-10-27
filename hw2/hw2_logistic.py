import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

w = np.load("model.npy")
mu = np.load("mu.npy")
sigma = np.load("sigma.npy")
b = np.load("b.npy")

def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return np.clip(res, 0.00000000000001, 0.999999999999)

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
#normalize for test data
for i,r in enumerate(test_x):
	test_x[i] = (test_x[i] - mu) / sigma
ans = []
id = 1
for r in test_x:
	ans.append([str(id)])
	#(1/pow((2 * math.pi), data_num/2 ))*(1/pow(np.linalg.det(variance),1/2)) 可消掉的項
	#print(math.exp((-1/2) * np.dot(np.dot(np.transpose(r - mean_class1) , inv(variance)) , (r - mean_class1))))
	z = np.dot(r,np.transpose(w)) + b
	y = sigmoid(z)
	if(y > 0.5):
		ans[id - 1].append(1)
	else:
		ans[id - 1].append(0)
	id = id + 1

text = open(sys.argv[6], "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()