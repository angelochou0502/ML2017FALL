import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

# read model
w = np.load('model.npy')
data_mean = np.load('data_mean.npy')
standard_deviation = np.load('standard_deviation.npy')

#read testing data
test_x = []
n_row = 0
cur_index = -1
text = open( sys.argv[1],"r" , encoding = 'big5')
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row % 18 == 0:
        n_row = 0
        cur_index = cur_index + 1
        test_x.append([])
        for i in range(2,11):
            test_x[cur_index].append((float(r[i]) -  data_mean[n_row]) / standard_deviation[n_row])
    else :
        #take the PM2.5 , PM 10 , O3 feature
        if n_row == 9 or n_row == 8 or n_row  == 7: #or n_row == 6 #or n_row == 5 or n_row == 4 or n_row == 3 or n_row == 2 or n_row == 1:
            for i in range(2,11):
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
    a = np.dot(w,test_x[i]) *  standard_deviation[9] + data_mean[9]
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()