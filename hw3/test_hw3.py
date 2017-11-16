import numpy as np
import csv
import sys
from keras.models import Sequential, load_model
from keras.layers import Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from tensorflow.python.client import device_lib
from keras.utils import np_utils

#load model
model_1 = load_model("model1")
model_2 = load_model("model2")

#read test data
test = open(sys.argv[1],'r',encoding = 'big5')
row = csv.reader(test)
next(row,None)
test_index = []
test_x = []
index = 0

#normalization
for r in row:
	test_index.append(int(r[0]))
	tmp = np.array([(float(k)/255) for k in r[1].split()])
	test_x.append([])
	test_x[index] = tmp.reshape(48,48,1)
	index = index + 1

test_index = np.array(test_index)
test_x = np.array(test_x)

#predict test data
ans = []
ans_index = 0
predict_1 = model_1.predict(test_x, batch_size = 100)
predict_2 = model_2.predict(test_x, batch_size = 100)
#predict_3 = model_3.predict(test_x, batch_size = 100)
predict_final = predict_1 + predict_2 

for i,r in enumerate(predict_1):
	ans.append([])
	ans[ans_index].append(test_index[i])
	ans[ans_index].append(predict_final[i].argmax())
	#ans[ans_index].append(r.argmax())
	ans_index = ans_index + 1



text = open(sys.argv[2], "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
