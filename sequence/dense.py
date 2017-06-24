import numpy as np 
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, normalization
# from keras.optimizers import SGD, RMSprop, Adam
from math import sqrt
import tensorflow as tf



def getData(filename):
	X = []
	Y = []
	fp = open(filename)
	A = fp.readlines()
	for i in range(1, len(A)):
		# print i
		B = A[i].strip()
		B = A[i].split('"')
		if(len(B) < 2):
			continue
		B = B[1].split(',')
		x = B[:len(B)-1]
		a =  B[len(B)-1]
		y = int(a)
		for i in range(len(x)):
			x[i] = int(x[i])
		X.append(x)
		Y.append([y])
	return X, Y

def processData(filename):
	X, Y = getData(filename)
	maxSeqLen = 0
	for i in range(len(X)):
		if(maxSeqLen < len(X[i])):
			maxSeqLen = len(X[i])

		# mean = sum(X[i])/len(X[i])
		# stddev = 0
		
		# for j in range(len(X[i])):
		# 	stddev += (X[i][j]- mean)*(X[i][j] - mean)
		# 	X[i][j] /= mean 

		# stddev = sqrt(stddev/len(X[i]))
		# for j in range(len(X[i])):
		# 	X[i][j] -= stddev

	for i in range(len(X)):
		l = maxSeqLen - len(X[i])
		s = [-1]*l
		X[i] = s+ X[i]


	return X, Y




# def train():

# 	X, Y = processData('train.csv')
# 	trainX = np.array(X)
# 	trainY = np.array(Y)
# 	# print trainX.shape , trainY.shape
# 	shape = trainX.shape[0]
# 	learning_rate = 0.001
# 	model = Sequential()
# 	model.add(Dense(units = 100, activation = 'relu', input_shape = (347,)))
# 	model.add(Dense(units = 50,activation='relu'))
# 	model.add(Dense(1, activation = 'linear'))                        
	               
# 	optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  

# 	normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)            
# 	# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)                          
# 	# optimizer=SGD(learning_rate,momentum=0.9,nesterov=True,clipnorm=1.)
# 	loss='mse'
# 	metrics=['mae', 'acc']
# 	model.compile(loss = 'mse', optimizer =optimizer, metrics = metrics)
# 	# model.compile(optimizer=optimizer,loss='mse',metrics=metrics)                        
# 	model.fit(trainX,trainY,epochs=20,batch_size=8)

# # processData('train.csv')



def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	# Output layer with linear activation
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	
	return out_layer



def train():
	X, Y = processData('train.csv')
	trainX = np.array(X)
	trainY = np.array(Y)
	input_shape = trainX.shape[1]
	num_examples = trainX.shape[0]

	# np.reshape(trainY, (num_examples, 1, 1))
	# print trainY.shape
	
	# print input_shape, num_examples
	learning_rate = 0.001
	training_epoch = 15
	batch_size = 64
	display_step = 20

	n_hidden_1 = 200 # 1st layer number of features
	n_hidden_2 = 100 # 2nd layer number of features
	n_input = input_shape 


	x = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, 1])


	weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
	}
	biases = {
	    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	    'out': tf.Variable(tf.random_normal([1]))
	}


	pred = multilayer_perceptron(x, weights, biases)

	cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions = pred))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess: 
		sess.run(init)

		for epoch in range(training_epoch):
			avg_cost = 0 
			total_batches = num_examples/batch_size

			for i in range(total_batches):
				batch_X = np.array(X[i*batch_size: (i+1)*batch_size])
				batch_Y = np.array(Y[i*batch_size: (i+1)*batch_size])

				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_X, y: batch_Y})

				avg_cost += c /total_batches

				if(epoch%display_step == 0):
					print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))



		print "Done!"




train()			
            