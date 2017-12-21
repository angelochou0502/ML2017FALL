import numpy as np
import sys
import keras
from keras.models import load_model
import keras.backend as K
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import train

def rmse(y_true , y_pred):
	return K.sqrt(K.mean(K.pow(y_pred - y_true , 2)))

def get_embedding(model):
	user_emb = np.array(model.layers[2].get_weights()).squeeze()
	print('user embedding shape:' , user_emb.shape)
	movie_emb = np.array(model.layers[3].get_weights()).squeeze()
	print('movie embedding shape: ' , movie_emb.shape)
	return user_emb , movie_emb

def general_type(type_data):
	Horror = ["Thriller" , "Horror"  , "Crime" , "Mystery" , "Film-Noir"]
	Romance = ["Drama" , "Musical" , "Romance"]
	Children = ["Children's" , "Animation" ]#, "Comedy"]
	Action = ["War" , "Action" , "Sci-Fi" , "Fantasy" , "Adventure" , "Documentary" , "Western"]
	Comedy = ["Comedy"]
	if(type_data in Horror):
		return 1
	elif(type_data in Romance):
		return 2
	elif(type_data in Children):
		return 3
	elif(type_data in Action):
		return 4
	elif(type_data in Comedy):
		return 5
	else:#no type should here 
		print("no fit type")

#movie type : 
# Horror : [thriller, horror , crime, mystery , film-noir]
# Romance : [drama , musical , romance]
# children : [children's , Animation , Comedy]
# Action : [War, Action , SciFi , Fantasy , Adventure , Documentary , Western]
def get_movie_type(fileName):
	Movie_ID = []
	Movie_Type = []
	#All_Type = set()
	with open(fileName , 'r' , encoding = 'latin-1') as f:
		lines = f.read().strip().split('\n')[1:]
		for line in lines:
			data = line.split('::')
			Movie_ID.append(int(data[0]))
			type_data = data[2].split('|')[0]
			Movie_Type.append(int(general_type(type_data)))
			#for types in type_data:
			#	All_Type.add(types)
	print(Movie_ID[1193])
	return Movie_ID ,  Movie_Type
		
def give_movie_type(Movie , Movie_ID , Movie_Type):
	Movie_Type_each = []
	for movie in Movie:
		Movie_Type_each.append(Movie_Type[Movie_ID.index(movie[0])])
	return Movie_Type_each

def embed_moive(Movie , movie_emb):
	Movie =  K.variable(Movie)
	e = keras.layers.Embedding(3953 , 256 ,  weights = [movie_emb])(Movie)
	e = keras.layers.Flatten()(e)
	#np.savetxt('testMed', keras.backend.eval(e)[:200], fmt='%.5f',delimiter=',')
	e = keras.backend.eval(e)
	return e

def draw(x,y):
	print("in draw")
	x = np.array(x , dtype = np.float64)
	y = np.array(y)
	x = x[:10000]
	y = y[:10000]
	print(x.shape)
	print(y.shape)
	#perform t-SNE embedding
	vis_data = TSNE(n_components=2).fit_transform(x)
	#plot the result
	vis_x = vis_data[: , 0]
	vis_y = vis_data[: , 1]
	#print("vis_data is " , vis_data)

	cm = plt.cm.get_cmap('RdYlBu')
	sc = plt.scatter(vis_x , vis_y , c=y , cmap=cm)
	plt.colorbar(sc)
	plt.show()

def main():
	model = keras.models.load_model("./weights.45-0.8562.h5" , custom_objects = {'rmse':rmse})
	#print(model.summary())
	user_emb , movie_emb= get_embedding(model)

	fileName = "data/train.csv"
	User , Movie  , rate , max_UID , max_MID = train.read_data(fileName)

	e = embed_moive(Movie , movie_emb)

	Movie_fileName = "data/movies.csv"
	Movie_ID , Movie_Type = get_movie_type(Movie_fileName)

	Movie_Type_each = give_movie_type(Movie , Movie_ID , Movie_Type)
	
	draw(e , Movie_Type_each)

if __name__ == "__main__":
	main()