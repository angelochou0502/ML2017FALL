import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

repeat = 100
l_rate = 1

#add batch_size
batch_size = 100

def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return np.clip(res, 0.00000000000001, 0.999999999999)

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

#normalization with feature 
mu = np.mean(feature, axis = 0)
sigma = np.std(feature , axis = 0)
for i,row in enumerate(feature):
	feature[i] = (feature[i] - mu) / sigma

w = np.zeros(len(feature[0])) #set number of feature which is 106
b = 0 # bias

train_x.close()
train_y.close()

#add adagrad
w_ada_tmp = np.zeros(len(feature[0]))
b_ada_tmp = 0

#regularization
lamda = 0.2

#training
for i in range(repeat):
	z = np.dot(feature,np.transpose(w)) + b
	y = sigmoid(z);
	cross_entropy = -(np.dot(label,np.log(y)) + np.dot((1 - label) , np.log(1 - y)))

	#regulization
	w_grad = np.sum(-1 * feature * (label - y).reshape((len(label), 1)), axis = 0) + lamda * np.sum(w ** 2)
	b_grad = np.sum(-1 * (label - y))
	#print(w_grad)
	#ada grad
	w_ada_tmp = w_ada_tmp + w_grad ** 2
	w_ada = np.sqrt(w_ada_tmp)
	b_ada_tmp = b_ada_tmp + b_grad ** 2
	b_ada = math.sqrt(b_ada_tmp)

	w = w - l_rate * w_grad / w_ada
	b = b - l_rate * b_grad / b_ada
	print ('iteration: %d | Cost: %f  ' % ( i ,cross_entropy))

#save model
np.save("model.npy", w)
np.save("mu.npy",mu)
np.save("sigma.npy",sigma)
np.save("b.npy",b)

w = np.load("model.npy")
mu = np.load("mu.npy")
sigma = np.load("sigma.npy")
b = np.load("b.npy")


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

filename = "result/logistic_sh_regu.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

