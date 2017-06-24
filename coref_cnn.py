# Maine kuch files bheje hai gmail pe
# Usko apne Downloads folder mein download karna
# Then open a terminal in that folder
# Then type the following commands:
# scp ./train_data_dev_X.npz 13CS30033@10.5.18.109:/home/du1/13CS30033/Houston
# scp ./train_data_dev_Y.npz 13CS30033@10.5.18.109:/home/du1/13CS30033/Houston
# scp ./cnn.py 13CS30033@10.5.18.109:/home/du1/13CS30033/Houston/cnn
# Then stop the running code
# Ek baar epoch number aur accuracy bata dena
# Then in that same screen, run the previous command again
# The command should be ~/coref/venv/bin/python cnn.py
# And tumne meko history waala dump nahi bheja hai
# Voh bhej dena

import pickle
import os 

Sentences = []
GloveVec = {}
IMAGE_SIZE = 50 
GloveVecSize = 300 ############ 

def getSentences():
	data_dir = "Coref_Data"
	for folder in os.lisdir(data_dir):
		path = os.path.join(data_dir, folder)
		fp = open(path+"smk_list.pickle")
		A = pickle.load(fp)
		Sentences = Sentences + A


def makeImage(phrase):
	words = phrase.split()
	l = len(words)

	image = []
	count = 0
	for i in range(IMAGE_SIZE):
		if(i < (IMAGE_SIZE - l)/2 ):
			image.append([0 for j in range(GloveVecSize)])
		elif(count < l):
			image.append(GloveVec[words[count]])
			count +=1
		else:
			image.append([0 for j in range(GloveVecSize)])

	print "Image created of size", len(image)
	return image



	



