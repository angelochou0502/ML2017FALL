import sys
import csv
import numpy as np
import keras
import keras.backend as K

def read_data(fileName):
	test_file = open(fileName , 'r')
	row = csv.reader(test_file)
	next(row, None)
	Data_id = []
	User = []
	Movie = []
	Gender = []
	Age = []

	#Age_ID = np.load("Age.npy")
	#Gender_ID = np.load("Gender.npy")
	#User_ID = np.load("User.npy")

	for r in row:
		Data_id.append(int(r[0]))
		User.append(int(r[1]))
		Movie.append(int(r[2]))

		#Gender.append(Gender_ID[np.where(User_ID == (int(r[1])))[0][0]])
		#Age.append(Age_ID[np.where(User_ID == (int(r[1])))[0][0]])
	return Data_id ,User, Movie , Age , Gender

def rmse(y_true , y_pred):
	return K.sqrt(K.mean(K.pow(y_pred - y_true , 2)))

def main():
	#fileName = "data/test.csv"
	Data_id , User , Movie , Age , Gender= read_data(sys.argv[1])

	model = keras.models.load_model("./weights.45-0.8562.h5" , custom_objects = {'rmse':rmse})

	User = np.array(User)
	Movie = np.array(Movie)
	Bias = Movie
	Age = np.array(Age)
	Gender = np.array(Gender)
	predict = model.predict([User , Movie  ] )
	#print(model.summary())

	ans = []
	ans_index = 0
	normalize = False
	#mean = np.load("rate_mean.npy")
	#std = np.load("rate_std.npy")
	for i,r in enumerate(predict):
		ans.append([])
		ans[ans_index].append(Data_id[i])
		if(normalize == False):
			if(predict[i][0] > 5 ):
				ans[ans_index].append(5 )
			elif(predict[i][0] < 1):
				ans[ans_index].append(1 )
			else:
				ans[ans_index].append(predict[i][0])
		else:
			#pred_ans = predict[i][0] * std + mean
			#pred_ans = predict[i][0] * 4 + 1
			if(pred_ans > 5 ):
				ans[ans_index].append(5)
			elif(pred_ans < 1):
				ans[ans_index].append(1)
			else:
				ans[ans_index].append(pred_ans)
		#ans[ans_index].append(r.argmax())
		ans_index = ans_index + 1

	#filename = "result/MF_256_05.csv"

	text = open(sys.argv[2], "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["TestDataID","Rating"])
	for i in range(len(ans)):
	    s.writerow(ans[i]) 
	text.close()


if __name__ == "__main__":
	main()
