import numpy as np
import csv
import sys
from keras import layers
from keras.models import Sequential,Model
from keras.layers import Flatten,LeakyReLU,Input
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.client import device_lib
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

epochs = 150
cardinality = 32

#load data
#data is one label and 48 * 48 intensity value for each
train = open(sys.argv[1],'r')
row = csv.reader(train)
next(row,None)


feature = []
label = []

#add normalize
now_row = 0
for r in row:
	label.append(float(r[0]))
	tmp = np.array([(float(k)/255) for k in r[1].split()])
	feature.append([])
	feature[now_row] = tmp.reshape(48,48,1)
	now_row = now_row + 1

feature = np.array(feature)
label = np.array(label)
label = np_utils.to_categorical(label)
data_num = len(feature)
split_num = int(data_num * 0.1)
train_x = feature[0: data_num - split_num , :]
train_y = label[0: data_num - split_num , :]
val_x = feature[data_num - split_num : (data_num - 1), :]
val_y = label[data_num - split_num : (data_num - 1) , :]

#data generator
datagen = ImageDataGenerator(rotation_range = 20 , width_shift_range = 0.2 , height_shift_range = 0.2 , horizontal_flip = True)
datagen.fit(feature)

#risidual network
def residual_block(y , nb_channels , _strides = (1,1) , _project_shortcut = False):
	shortcut = y
	y = layers.Conv2D(nb_channels , kernel_size = (3,3) , strides = _strides , padding = 'same')(y)
	y = layers.BatchNormalization()(y)
	y = layers.LeakyReLU()(y)

	y = layers.Conv2D(nb_channels , kernel_size = (3,3) , strides = (1,1) , padding = 'same')(y)
	y = layers.BatchNormalization()(y)

	# identity shortcuts used directly when the input and output are of the same dimensions
	#if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
	shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
	shortcut = layers.BatchNormalization()(shortcut)
	y = layers.add([shortcut, y])
	y = layers.LeakyReLU()(y)
	return y

#training
input_img = Input(shape=(48, 48, 1))

block1 = Conv2D(64, (3,3) , padding = 'same' , activation = 'relu' )(input_img)
block1 = MaxPooling2D((2,2))(block1)
#block1 = MaxPooling2D(pool_size = (3,3) , strides = (2,2))(block1)

block2 = residual_block(block1, 64)
block3 = residual_block(block2, 64)
block4 = residual_block(block3, 64)
block4 = MaxPooling2D((2,2))(block4)

block5 = residual_block(block4, 128)
block6 = residual_block(block5, 128)
block7 = residual_block(block6, 128)
block7 = MaxPooling2D((2,2))(block7)

block8 = residual_block(block7, 512)
block9 = residual_block(block8, 512)

block10 = Conv2D(512,(3,3) , activation = 'relu')(block9)
block10 = layers.BatchNormalization()(block10)
block10 = AveragePooling2D(pool_size=(3,3))(block10)
block10 = Flatten()(block10)

fc1 = Dense(1024, activation='relu')(block10)
fc1 = Dropout(0.5)(fc1)

fc2 = Dense(1024, activation='relu')(fc1)
fc2 = Dropout(0.5)(fc2)

predict = Dense(7)(fc2)
predict = Activation('softmax')(predict)
model = Model(inputs=input_img, outputs=predict)

model.compile(loss = 'categorical_crossentropy' , optimizer = Adam(lr=1e-3) , metrics = ['accuracy'])
#model.summary()

#for the submission submit out the callback to make it faster
'''chCallBack = ModelCheckpoint('./residual_augment_10block_changefilternum/weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
esCallBack = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')
tbCallBack = TensorBoard(log_dir='residual_augment_10block_changefilternum/Graph3', histogram_freq=1, write_graph=True, write_images=True)'''

model.fit_generator(datagen.flow(train_x,train_y, batch_size = 32) , steps_per_epoch = len(train_x) / 32 , epochs = epochs ,\
	                validation_data = (val_x , val_y) ,shuffle = True)



result = model.evaluate(feature, label , batch_size = 100) 
print('\nTrain Acc:' , result[1])

print(model.summary())

model.save('residual_augment_10block.h5')






'''
for e in range(epochs):
	print('epoch' , e)
	batches = 0
	for x_batch, y_batch in datagen.flow(feature, label , batch_size = 32):
		print("the total of train this round")
		print(len(x_batch))
		model.fit(x_batch , y_batch)
		batches += 1
		if batches >= len(feature) / 32 :
			break;
'''