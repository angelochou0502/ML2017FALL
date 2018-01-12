import numpy as np
import sys
import skimage
import skimage.io
from numpy.linalg import eig , svd

def loadImage(fileName):
	image = np.load(fileName)
	#image -= np.min(image)
	#image /= np.max(image)
	#image = (image * 255).astype(np.unit8)
	return image

def cov(image):
	return np.dot(image.T , image) / image.shape[0]

def pca(image , count):
	image -= np.mean(image)
	image /= np.std(image)
	C = cov(image)
	eig_values , eig_vectors = eig(C)
	key = np.argsort(eig_values)[::-1][:count]
	E , V = eig_values[key] , eig_vectors[: , key]
	print(E)
	return E, V

def showImage(image , fileName):
	image -= np.min(image)
	image /= np.max(image)
	image = (image*255).astype('uint8')
	skimage.io.imsave(fileName , image)
	#skimage.io.imshow(image)
	#skimage.io.show()

def draw_mean(images):
	mean_image = np.mean(images , axis = 0)
	#print(mean_image.shape)
	showImage(mean_image)
	return;

def get_and_show_eigenface(data_mean):
	data_mean = data_mean.reshape((data_mean.shape[0] , -1))
	u ,s , v = svd(data_mean,0)
	#draw the previous four eigenface
	eigenface = v[0:4]
	eigenvalue = s[0:4]
	eigenface = eigenface.reshape((4,600,600,3))
	#np.save('eigenface_top4.npy', eigenface)
	#np.save('eigenvalue_top4.npy',eigenvalue)
	return eigenface

def reconstruct(eigenface , data , mean_image , target_image):
	data_mean = data - mean_image
	#choose_img = data_mean[number]
	choose_img = target_image - mean_image;
	x_construct = 0
	for i in range(4): # 4 eigenface
		x_reduced = np.dot(choose_img.reshape(-1) , eigenface[i].reshape(-1))
		x_construct += x_reduced * eigenface[i]
	x_construct += mean_image
	showImage(x_construct , './reconstrction.jpg')
	return;

def main():
	fileName = "data/Aberdeen"
	fileName = sys.argv[1]
	images = skimage.io.imread_collection(fileName + "/*.jpg")
	data = []
	for image in images:
		data.append(image)
	data = np.array(data)

	#draw image mean -> for report 1-1
	#draw_mean(data)

	mean_image = np.mean(data , axis = 0)
	data_mean = data - mean_image

	eigenface = get_and_show_eigenface(data_mean)
	#eigenface = np.load('eigenface_top4.npy')

	#reconstruct through eigenface
	targetFile = sys.argv[1] + '/' + sys.argv[2]
	target_image = skimage.io.imread(targetFile)
	reconstruct(eigenface , data , mean_image , target_image)


	#for image in images:
	#	print(image.shape)
	#skimage.io.imshow(image)
	#skimage.io.show()

	#E_r , V_r = pca(image_r , 4)
	#E_g , V_g = pca(image_g , 4)
	#E_b , V_b = pca(image_b , 4)


if __name__ == "__main__":
	main()