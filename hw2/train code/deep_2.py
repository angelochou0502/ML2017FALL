import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
import csv 
import sys
from tensorflow.python.client import device_lib


#load data
train_x = open('data/X_train','r', encoding='big5')
train_y = open('data/Y_train','r', encoding='big5')

row_x = csv.reader(train_x)
next(row_x,None) #skip the first line
row_y = csv.reader(train_y)
next(row_y,None) #skip the first line

#get out label and feature data
feature = [];
label = [];

#column one and two stands for class one and zero
for i,r in enumerate(row_y):
	label.append([])
	num = float(r[0])
	if(num == 0):
		label[i].append(1)
		label[i].append(0)
	else:
		label[i].append(0)
		label[i].append(1)

now_row = 0
for r in row_x:
	feature.append([])
	for i in range(len(r)):
		feature[now_row].append(float(r[i]))
	now_row = now_row + 1

feature = np.array(feature)
label = np.array(label)


#normalization with feature 
#mu = np.mean(feature, axis = 0)
#sigma = np.std(feature , axis = 0)

mu = np.load("mu.npy")
sigma = np.load("sigma.npy")

for i,row in enumerate(feature):
	feature[i] = (feature[i] - mu) / sigma

train_x.close()
train_y.close()



#deep learning

model = Sequential()

model.add(Dense(input_dim = len(feature[0]) , units = 600 , activation = 'sigmoid'))
model.add(Dense(units = 600 , activation = 'sigmoid'))

model.add(Dense(units = 2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

model.fit(feature , label , batch_size = 100 , epochs = 20)

result = model.evaluate(feature, label , batch_size = 100) 
print('\nTrain Acc:' , result[1]) 

model.save('my_model.h5')
#read test data
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

#predict test data
ans = []
predict = model.predict(test_x, batch_size = 100)
for i,r in enumerate(predict):
	ans.append([str(i+1)])
	if(r[0] > r[1]):
		ans[i].append(0)
	else:
		ans[i].append(1)


filename = "result/deep_sigmoid_2layer.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
