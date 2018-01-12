import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,SpectralClustering
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.manifold import TSNE
import keras
from keras.layers import Input , Dense
from keras.models import Model
import sys

def pca_reduction(images , n_components):
	pca = PCA(n_components = n_components)
	reduc_Image = pca.fit_transform(images)
	inverse_Image =  pca.inverse_transform(reduc_Image)
	return reduc_Image,inverse_Image

def TSNE_reduction(images , n_components):
	tsne = TSNE(n_components = n_components)
	reduc_Image = tsne.fit_transform(images)
	print(reduc_Image.shape)
	np.save("TSNE_pca48.npy" , reduc_Image)
	return reduc_Image

def do_kmeans(reduc_Image , n_clusters):
	kmeans = KMeans(n_clusters = n_clusters)
	#cluster = kmeans.fit_predict(reduc_Image)
	cluster = kmeans.fit(reduc_Image).labels_
	return cluster;

def read_test(filename):
	test_id = []
	index1 = []
	index2 = []

	test_file = open(filename , 'r')
	row = csv.reader(test_file)
	next(row, None)

	for r in row:
		test_id.append(int(r[0]))
		index1.append(int(r[1]))
		index2.append(int(r[2]))

	return test_id , index1 , index2

def write_predict_cluster(filename , test_id , index1 , index2 , cluster):
	ans = []
	ans_index = 0

	for i in range(np.array(test_id).shape[0]):
		ans.append([])
		ans[i].append(test_id[i])
		if( (cluster[index1[i]]) == (cluster[index2[i]])):
			ans[i].append(1)
		else:
			ans[i].append(0)


	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["ID","Ans"])
	for i in range(len(ans)):
	    s.writerow(ans[i]) 
	text.close()

def write_predict_similarity(filename , test_id , index1 , index2 ,reduc_Image , threshold , method):
	ans = []
	ans_index = 0
	one_num = 0
	zero_num = 0
	distance = []

	for i in range(np.array(test_id).shape[0]):
		ans.append([])
		ans[i].append(test_id[i])
		if(method == "cosine_similarity"):
			similarity = 1 - spatial.distance.cosine(reduc_Image[index1[i]], reduc_Image[index2[i]])
			if(similarity > threshold):
				ans[i].append(1)
				one_num = one_num + 1

			else:
				ans[i].append(0)
				zero_num = zero_num + 1
		elif(method == "euclidean"):
			distance.append(np.sum((reduc_Image[index1[i]] - reduc_Image[index2[i]]) ** 2))
	if(method == "euclidean"):
		distance = np.array(distance)
		normed_distance = (distance - distance.mean()) / distance.std()
		for i in range(np.array(test_id).shape[0]):
			if(normed_distance[i] > 0 ):
				ans[i].append(0)
				zero_num = zero_num + 1
			else:
				ans[i].append(1)
				one_num = one_num + 1

	print('one_num is %d , and zero_num is %d'  %(one_num , zero_num))
	

	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["ID","Ans"])
	for i in range(len(ans)):
	    s.writerow(ans[i]) 
	text.close()


def showImages(image1 , image2):
	fig = plt.figure()
	a=fig.add_subplot(1,2,1)
	plt.imshow(image1)
	a=fig.add_subplot(1,2,2)
	plt.imshow(image2)
	plt.show()
	quit()
	return;

def show_previous_25pic(inverse_Image ,images):
	fig = plt.figure()
	for i in range(5):
		for j in range(5):
			a = fig.add_subplot(5,5,5*i+j+1)
			plt.imshow(images[5*i+j].reshape((28,28)))

	fig2 = plt.figure()
	for i in range(5):
		for j in range(5):
			a2 = fig2.add_subplot(5,5,5*i+j+1)
			plt.imshow(inverse_Image[5*i+j].reshape((28,28)))
	plt.show()
	quit()
	return;

def draw_encode_decode(n , encoder , decoder , images):
	encoded_imgs = encoder.predict(images)
	decoded_imgs = decoder.predict(encoded_imgs)
	plt.figure(figsize=(20, 4))
	for i in range(n):
		# display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(images[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(decoded_imgs[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

def dnn_autoencoder(encoding_dim):
	input_img = Input(shape = (784,))
	encoded = Dense(128 , activation = 'relu')(input_img)
	encoded = Dense(64 , activation = 'relu')(encoded)
	encoder_output = Dense(encoding_dim)(encoded)

	decoded = Dense(64 , activation = 'relu')(encoder_output)
	decoded = Dense(128 , activation = 'relu')(decoded)
	decoded = Dense(784 , activation = 'tanh')(decoded)
	autoencoder =  Model(input_img , decoded)
	
	encoder = Model(input_img , encoder_output)
	return autoencoder , encoder

def Do_DnnAutoencoder(images , encoding_dim):
	images = images.astype('float32') / 255.

	'''
	autoencoder , encoder  = dnn_autoencoder(encoding_dim)
	autoencoder.compile(optimizer = 'adam' , loss = 'mse')
	esCallBack = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto')
	autoencoder.fit(images , images , validation_split = 0.1 , epochs = 400 , batch_size = 128 , shuffle = True , callbacks = [esCallBack])
	encoder.save('encoder/mophan_deep_dim32_400epochs_128batchsize.h5')
	'''
	encoder = keras.models.load_model('encoder.h5')

	reduc_Image = encoder.predict(images)
	#decoder.save('encoder/deep_dim32_decoder.h5')
	#draw_encode_decode(20 , encoder , decoder , images)
	return reduc_Image

def visulize():
	#images = np.load('data/visualization.npy')
	#encoded_imges = Do_DnnAutoencoder(images , 32)
	#x_embedded = TSNE(n_components = 2).fit_transform(encoded_imges)
	#np.save('x_embedded.npy' , x_embedded)
	x_embedded = np.load('x_embedded.npy')

	n_clusters = 2
	cluster = do_kmeans(x_embedded , n_clusters)
	'''	
	for i in range(10000):
		if(cluster[i] == 0):
			plt.scatter(x_embedded[i , 0] , x_embedded[i , 1] , c = 'r' , s = 0.2)
		else:
			plt.scatter(x_embedded[i , 0] , x_embedded[i , 1] , c = 'b' , s = 0.2)
	plt.savefig('final_kmeans_myself.png')
	quit()
	'''
	plt.scatter(x_embedded[:5000, 0] , x_embedded[:5000,1] , c = 'b' , label = 'dataset A' , s = 0.2)
	plt.scatter(x_embedded[5000:, 0] , x_embedded[5000:,1] , c = 'r' , label = 'dataset B' , s = 0.2)
	plt.savefig('final_tsne.png')
	quit()


def main():
	#images is a 140000 * 784(28*28) matrix
	#images_file = sys.argv[1]
	#images = np.load(images_file)

	#see the image data
	#np.set_printoptions(threshold = np.nan)
	#np.savetxt("image.txt" , images[0:20])

	#DNN autoendoer
	#encoding_dim = 32
	#reduc_Image = Do_DnnAutoencoder(images , encoding_dim)
	#decoder = keras.models.load_model('encoder/dim32_decoder.h5')

	'''
	#dimesion reduction (PCA)
	n_components = 48
	reduc_Image , inverse_Image = pca_reduction(images , n_components)
	#reduc_Image = TSNE_reduction(reduc_Image , 2)
	#show_previous_25pic(inverse_Image , images)
	TSNE_autoencoder = np.load("TSNE_pca48.npy")
	plt.plot(TSNE_autoencoder[0:5000,0] , TSNE_autoencoder[0:5000,1] , 'ro')
	plt.show()
	'''

	#clustering -> 
	#K-means
	#n_clusters = 2
	#cluster = do_kmeans(reduc_Image , n_clusters)
	#np.save("cluster.npy",cluster)
	cluster = np.load("cluster.npy")

	#read test data
	filename = "data/test_case.csv"
	test_file = sys.argv[2]
	test_id , index1 , index2 = read_test(test_file)

	#pridict with the pixel count
	#filename = "result/pixelCount230.csv"
	#write_predict_cluster(filename , test_id , index1 , index2 , pixelCount)

	#predict and write to result file
	out_file = sys.argv[3]
	write_predict_cluster(out_file , test_id , index1 , index2 , cluster)

	#threshold = -0.1
	#method : euclidean or cosine_similarity
	#method = "euclidean"
	#filename = "result/pca{%d}_%s{%.1f}.csv" %(n_components, method ,threshold)
	#write_predict_similarity(filename , test_id , index1 , index2 , reduc_Image , threshold , method )


if __name__ == "__main__":
	main()