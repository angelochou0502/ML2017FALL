import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

#read data
data = []
#每一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])



n_row = 0
text = open('data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0)) 
    n_row = n_row + 1
text.close()

#feature scaling count the mean and standard deviation
#new_pm2.5 = (pm2.5 - data_mean[9]) / standard_deviation[9]
if_feature_scal = True
normalization = True
unity_normalization = False
if(if_feature_scal and normalization):
    standard_deviation = []
    data_mean = np.sum(data, axis = 1) / len(data[0])
    for i,row in enumerate(data):
        standard_deviation.append(np.sqrt(np.sum((row - data_mean[i]) ** 2) / len(data[0])))

    for i,row in enumerate(data):
        data[i] = (row - data_mean[i]) / standard_deviation[i]

mother = np.max(data[9]) - np.min(data[9])
son = np.min(data[9])
min_of_data = []
max_of_data = []
if(if_feature_scal and unity_normalization):
    for i,row in enumerate(data):
        min_of_data.append(np.min(data[i]))
        max_of_data.append(np.max(data[i]))
        data[i] = (row - np.min(data[i])) / (np.max(data[i]) - np.min(data[i])) 
        
#parse data to (x,y)
x = []
y = []
# num of data count before
data_num = 9
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(480 - data_num):
        x.append([])
        # 18種污染物
        for t in range(18):
            #取PM2.5 , PM10 , O3 , CO , NMHC, NOx 當參數
            if t == 9 or t == 8 or t == 7 or t == 0:#t == 5 or t == 4 or t == 3 or t ==2 or t == 1 or t == 0: # t == 2:
                # 連續9小時
                for s in range(data_num):
                    x[(480- data_num) *i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+data_num] )
x = np.array(x)
y = np.array(y)

#regulization
lamda = 0.1

#讓資料中無負數值
'''
min_data = x.min()
x = x - min_data
y = y - min_data
'''

#feature scaling back to normal size
if(if_feature_scal and normalization):
    y = y  * standard_deviation[9] + data_mean[9] 
if(if_feature_scal and unity_normalization):
    y = y * mother + son

# add square term
x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0]))
l_rate = 20
repeat = 20000

#start training
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    #feature scaling back to normal size
    if(if_feature_scal and normalization):
        hypo = hypo  * standard_deviation[9]  + data_mean[9] 
    if(if_feature_scal and unity_normalization):
        hypo = hypo * mother - son
    loss = hypo - y + lamda * np.sum(w ** 2)
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

# save and read model
# save model
np.save('model.npy',w)
np.save('data_mean.npy',data_mean)
np.save('standard_deviation.npy',standard_deviation)
# read model
w = np.load('model.npy')
data_mean = np.load('data_mean.npy')
standard_deviation = np.load('standard_deviation.npy')

#read testing data
test_x = []
n_row = 0
cur_index = -1
text = open('data/test.csv' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row % 18 == 0:
        n_row = 0
        cur_index = cur_index + 1
        test_x.append([])
        for i in range(11 - data_num,11):
            test_x[cur_index].append((float(r[i]) -  data_mean[n_row]) / standard_deviation[n_row] )
    else :
        #take the PM2.5 , PM 10 , O3 feature
        if n_row == 9 or n_row == 8 or n_row  == 7: #or n_row == 6 #or n_row == 5 or n_row == 4 or n_row == 3 or n_row == 2 or n_row == 1:
            for i in range(11 - data_num,11):
                if r[i] !="NR":
                    test_x[cur_index].append((float(r[i]) -  data_mean[n_row]) / standard_deviation[n_row])
                else:
                    test_x[cur_index].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

'''
for index,row in enumerate(test_x):
    print(len(row))
'''

# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

#get ans with your model
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i]) * standard_deviation[9] + data_mean[9]
    ans[i].append(a)

filename = "result/best_feature_regulization_0789_testfeature_squareterm.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()