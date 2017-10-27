import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
import csv 
import sys
from tensorflow.python.client import device_lib

#load model
model = load_model("my_model.h5")
mu = np.load("mu.npy")
sigma = np.load("sigma.npy")

#read test data
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

#predict test data
ans = []
predict = model.predict(test_x, batch_size = 100)
for i,r in enumerate(predict):
	ans.append([str(i+1)])
	if(r[0] > r[1]):
		ans[i].append(0)
	else:
		ans[i].append(1)


text = open(sys.argv[6], "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()