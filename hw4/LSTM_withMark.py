import numpy as np
from numpy import asarray
import pandas as pd
import json
import sys
import re
import pickle
from keras.models import Sequential
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

def readfile(fileName):
	read_data = pd.read_csv(fileName, sep = '\+\+\+\$\+\+\+', header=None , engine = 'python')
	#print(read_data.loc[:,1])
	#print(read_data.shape)
	return read_data

def sep_file(fileName):
	docs = []
	train_y = np.zeros((read_data.shape[0] , 2))
	for i in range(read_data.shape[0]):
		label = read_data.loc[:,0][i].astype(int);
		if(label == 0):
			train_y[i] = np.array([1,0])
		else:
			train_y[i] = np.array([0,1])
		docs.append(read_data.loc[:,1][i])
	t = Tokenizer(filters = '\n')
	t.fit_on_texts(docs)
	vocab_size = len(t.word_index) + 1
	encoded_docs = t.texts_to_sequences(docs)
	max_len = len(max(encoded_docs , key = len ))
	padded_docs = pad_sequences(encoded_docs , maxlen = max_len , padding = 'post')
	pickle.dump(t , open('my_tokenizer.p', 'wb'))
	np.save('max_len.npy',max_len)
	return vocab_size , t , train_y , max_len , padded_docs

def get_wordVec(fileName):
	embeddings_index = dict()
	f = open(fileName)
	next(f)
	for line in f:
		values = line.split()
		word = values[0]
		coefs = asarray(values[1:] , dtype = 'float32')
		embeddings_index[word] = coefs
	f.close()
	np.save("my_dict_nolabel_withMark.npy",embeddings_index)

def create_matrix(my_dict , vocab_size , t):
	embedding_matrix = np.zeros((vocab_size , 200))
	for word, i in t.word_index.items():
		embedding_vector = my_dict.tolist().get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	return embedding_matrix
	
def model( vocab_size , embedding_matrix , max_len):
	#training
	model = Sequential();
	e = Embedding(vocab_size , 200 , weights = [embedding_matrix] , input_length = max_len , trainable=False)
	model.add(e)
	model.add(LSTM(units = 1000, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, \
				kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', \
				unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, \
				activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, \
				dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, \
				go_backwards=False, stateful=False, unroll=False))
	model.add(Dense(2))
	model.add(Activation('softmax'))
	model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
	return model

if __name__ == "__main__":
	fileName = sys.argv[1]
	read_data = readfile(fileName)

	#get the dictionary only do one time , to make the my_dict_nolabel_withMark.npy
	#get_wordVec('training/my.cbow.200d.withmark.txt') 

	my_dict = np.load("training/my_dict_nolabel_withMark.npy")


	vocab_size , t , train_y , max_len , padded_docs= sep_file(fileName)

	embedding_matrix = create_matrix(my_dict , vocab_size , t)


	model = model( vocab_size , embedding_matrix , max_len);

	#chCallBack = ModelCheckpoint('./LSTM_1000unit_3layer/weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	model.fit(padded_docs , train_y , batch_size = 500 , epochs = 10 ,validation_split = 0.1 , shuffle = True )
	result = model.evaluate(padded_docs, train_y , batch_size = 100) 

	model.save('withMark.h5')
