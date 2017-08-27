from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save  # hint to help gc free up memory
	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	# Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

batch_size = 128
sgd_graph = tf.Graph()
layer1_weight_size = 20
layer2_weight_size = 10
F_D = []
with sgd_graph.as_default():
	#now all the training data into placeholder  which will be fed actual data at every call of session.run()
	tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	weights = tf.Variable(tf.truncated_normal([image_size * image_size, layer1_weight_size]))
	biases = tf.Variable(tf.zeros([layer1_weight_size]))
	weights_2 = tf.Variable(tf.truncated_normal([layer1_weight_size, layer2_weight_size]))
	biases_2 = tf.Variable(tf.zeros([layer2_weight_size]))
	weights_3 = tf.Variable(tf.truncated_normal([layer2_weight_size, num_labels]))
	biases_3 = tf.Variable(tf.zeros([num_labels]))
	global_ = tf.Variable(tf.constant(0))  
	learning_rate = 0.1
	decay_rate = 0.96  
	decay_steps = 100 
	
	def model(data):
		#define training computation , hidden layer + l2 regularization
		layer1 = tf.matmul(data, weights) + biases
		#layer1 = tf.nn.dropout(layer1, keep_prob=0.5)
		#hidden = tf.nn.relu(layer1)
		layer2 = tf.matmul(layer1, weights_2) + biases_2
		#layer2 = tf.nn.dropout(layer2, keep_prob=0.5)
		layer3 = tf.matmul(layer2, weights_3) + biases_3
		return layer3
	
	#cross_entropy between groud truth label and logistic predicts labels
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=model(tf_train_dataset)))
	regularizers = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)+tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(biases_2)+tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(biases_3)
	#define optimizer, using gradient descent and adaptive learning rate
	my_learning_rate = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)  
	optimizer = tf.train.GradientDescentOptimizer(my_learning_rate).minimize(loss+5e-4*regularizers)
	
	#define accuracy, not part of training, but let us watch the accuracy figures as we train
	#JST: compare the hypthesis between training, validation and test set
	#weights,biases are our training result
	train_prediction = tf.nn.softmax(model(tf_train_dataset))
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 3001
with tf.Session(graph = sgd_graph) as jst:
	tf.global_variables_initializer().run()
	print("Initialized")
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		# Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		# Prepare a dictionary telling the session where to feed the minibatch.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, global_ : step}
		_, l, rate ,predictions = jst.run([optimizer, loss, my_learning_rate, train_prediction], feed_dict=feed_dict)
		F_D.append(rate) 
		if (step % 100 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
			print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
			print('learning_rate:%.8f' % rate)
plt.figure(1)  
plt.plot(range(num_steps), F_D, 'r-') 
plt.show()   

