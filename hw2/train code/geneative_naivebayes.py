import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import scipy.stats 

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
'''
feature note : 
2 : sex 
5~14 : workclass

'''
#seperate data of different class
#label 0 stands for < 50k
train_class1_con = [] # < 50 k 
train_class1_dis = []
index_class1 = 0

train_class2_con = [] # >= 50k 
train_class2_dis = []
index_class2 = 0

feature_num = 106
for i in range(len(label)):
	if(label[i] == 0):
		train_class1_con.append([])
		train_class1_dis.append([])
		for j in range(feature_num):
			if(j == 0 or j == 1 or j == 3 or j == 4 or j == 5):
				train_class1_con[index_class1].append(feature[i][j])
			else:
				train_class1_dis[index_class1].append(feature[i][j])
		index_class1 = index_class1 + 1
	elif(label[i] == 1):
		train_class2_con.append([])
		train_class2_dis.append([])
		for j in range(feature_num):
			if(j == 0 or j == 1 or j == 3 or j == 4 or j == 5):
				train_class2_con[index_class2].append(feature[i][j])
			else:
				train_class2_dis[index_class2].append(feature[i][j])
		index_class2 = index_class2 + 1

train_class1_con = np.array(train_class1_con)
train_class1_dis = np.array(train_class1_dis)
train_class2_con = np.array(train_class2_con)
train_class2_dis = np.array(train_class2_dis)

#feature 0 , 1 , 3 , 4 , 5 is continuous, others are discrete

#caculate for continuous feature
mean_class1 = []
mean_class2 = []
variance_class1 = np.zeros( len(train_class1_con[0]) ) #only 5 features is continuous data
variance_class2 = np.zeros( len(train_class2_con[0]) )
variance = []

for i in np.transpose(train_class1_con):
	mean_class1.append(np.mean(i))
for i in np.transpose(train_class2_con):
	mean_class2.append(np.mean(i))

mean_class1 = np.array(mean_class1)
mean_class2 = np.array(mean_class2)

for i in range(len(train_class1_con)):
	variance_class1 = variance_class1 + (np.array([train_class1_con[i] - mean_class1]) ** 2)
variance_class1 = variance_class1 / len(train_class1_con)
for i in range(len(train_class2_con)):
	variance_class2 = variance_class2 +(np.array([train_class2_con[i] - mean_class2]) ** 2)
variance_class2 = variance_class2 / len(train_class2_con)

#to prove the correctness of my own caculation of variance
'''
print(variance_class1)
print(np.std(train_class1_con , axis = 0) ** 2)
quit()
'''

total_data = len(train_class1_con) + len(train_class2_con)
class1_num = len(train_class1_con)
class2_num = len(train_class2_con)
variance = float(class1_num) * variance_class1 / total_data + float(class2_num) * variance_class2 / total_data
variance = np.array(variance)

#data_num = len(train_class1_con[0])

#caculate for discrete feature
sum_class1 = []
sum_class2 = []
prob_class1 = []
prob_class2 = []
for i in np.transpose(train_class1_dis):
	sum_class1.append(np.sum(i))
for i in np.transpose(train_class2_dis):
	sum_class2.append(np.sum(i))
sum_class1 = np.array(sum_class1)
sum_class2 = np.array(sum_class2)

#probability of 1
prob_class1 = sum_class1 / len(train_class1_dis)
prob_class2 = sum_class2 / len(train_class2_dis)

'''
discrete note:
0 : sex
1~9 : workclass
10~ 25 : education
26~ 32 : martial status
33~ 47 : occupation
48 ~ 53 : relationship
54 ~ 58 : race
59 ~ 100 : native-country

#validate formua
count = 0
for i  in range(59, 101):
	print(prob_class1[i])
	count = count + prob_class1[i]
print(count)
quit()
'''
workclass = np.array(range(1,10))
education = np.array(range(10,26))
martial_status = np.array(range(26,33))
occupation = np.array(range(33,48))
relationship = np.array(range(48,54))
race = np.array(range(54,59))
native_country = np.array(range(59,101))

#faster way to calculate mean
'''
print(prob_class1)
print(np.mean(train_class1_dis, axis = 0))
quit()
'''


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


ans_index = 0
for n in test_x:
	ans.append([])
	con_index = 0
	dis_index = 0
	p_1 = float(1)
	p_2 = float(1)
	for j in range(feature_num):
		if(j == 0 or j == 1 or j == 3 or j == 4 or j == 5):
			if(j == 5 ):
				continue
			p_1 = p_1 * pow(2 * math.pi * variance_class1[0][con_index] ,- 1/2) * math.exp(-pow(float(n[j])-mean_class1[con_index],2)/(2 * variance_class1[0][con_index] ))
			p_2 = p_2 * pow(2 * math.pi * variance_class2[0][con_index] ,- 1/2) * math.exp(-pow(float(n[j])-mean_class2[con_index],2)/(2 * variance_class2[0][con_index] ))
			#faster way to do the gaussian caculate
			#print(scipy.stats.norm(mean_class1[con_index] , math.sqrt(variance_class1[0][con_index])).pdf(n[j]))
			con_index = con_index + 1
		else:
			if(((native_country == dis_index).any() or (race == dis_index).any() or dis_index == 0 or (occupation == dis_index).any()) == False):
				if(float(n[j]) == 1):
					p_1 = p_1 * prob_class1[dis_index]
					p_2 = p_2 * prob_class2[dis_index]
				elif(float(n[j]) == 0):
					p_1 = p_1 * (1 - prob_class1[dis_index])
					p_2 = p_2 * (1 - prob_class2[dis_index])
			dis_index = dis_index + 1
	if(p_1 >= p_2):
		ans[ans_index].append(str(ans_index+1))
		ans[ans_index].append(0)
	else:
		ans[ans_index].append(str(ans_index+1))
		ans[ans_index].append(1)
	ans_index = ans_index + 1




filename = "result/generative_naivebayes4_norace&natcountry&workhours&sex&occupation.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

