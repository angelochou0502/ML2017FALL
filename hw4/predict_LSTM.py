import numpy as np
from numpy import asarray
import pandas as pd
import json
import sys
import re
import csv
import pickle
from keras.models import Sequential,load_model
from keras.layers import Flatten, LSTM, TimeDistributed
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.client import device_lib
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer

def sep_file(fileName):
	docs = []
	test_index = []
	with open(fileName , encoding = 'utf-8') as fp:
		next(fp)
		for line in fp:
			test_index.append(int(line.split(",",1)[0]))
			docs.append(line.split(",",1)[1])

	t = pickle.load(open('my_tokenizer.p' , 'rb'))
	max_len = np.load('max_len.npy')
	encoded_docs = t.texts_to_sequences(docs)
	padded_docs = pad_sequences(encoded_docs , maxlen = max_len , padding = 'post')

	return padded_docs,test_index

def sep_file_withMark(fileName):
	docs = []
	test_index = []
	with open(fileName , encoding = 'utf-8') as fp:
		next(fp)
		for line in fp:
			test_index.append(int(line.split(",",1)[0]))
			docs.append(line.split(",",1)[1])

	t = pickle.load(open('my_tokenizer_withMark.p' , 'rb'))
	max_len = np.load('max_len_withMark.npy')
	encoded_docs = t.texts_to_sequences(docs)
	padded_docs = pad_sequences(encoded_docs , maxlen = max_len , padding = 'post')

	return padded_docs,test_index

def sep_file_BOW(fileName):
	docs = []
	test_index = []
	with open(fileName , encoding = 'utf-8') as fp:
		next(fp)
		for line in fp:
			test_index.append(int(line.split(",",1)[0]))
			docs.append(line.split(",",1)[1])

	t = pickle.load(open('my_tokenizer_BOW.p' , 'rb'))
	padded_docs = t.texts_to_matrix(docs , 'count')

	return padded_docs,test_index


if __name__ == "__main__":
	fileName = sys.argv[1]
	#get_dictionary(read_data) # only do one time
	#my_dict = np.load("my_dict.npy").tolist()
	#dict_len = np.load("dict_len.npy")

	#get the dictionary only do one time
	#get_wordVec('data/my.cbow.200d.txt') 

	### select what model we want here ###
	#padded_docs , test_index = sep_file(fileName)
	padded_docs , test_index = sep_file_withMark(fileName)
	#padded_docs , test_index = sep_file_BOW(fileName)


	model = load_model("withMark.h5")

	predict = model.predict(padded_docs , batch_size = 100)
	ans = []
	ans_index = 0
	for i,r in enumerate(predict):
		ans.append([])
		ans[ans_index].append(test_index[i])
		ans[ans_index].append(predict[i].argmax())
		#ans[ans_index].append(r.argmax())
		ans_index = ans_index + 1


	filename = sys.argv[2]
	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["id","label"])
	for i in range(len(ans)):
	    s.writerow(ans[i]) 
	text.close()