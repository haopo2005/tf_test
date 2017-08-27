from __future__ import print_function #__future__ , make python2.7 has the print() function in python3.0+
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

#load data set
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	
	del save # gc to free up memory
	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)

'''


Reformat into a shape that's more adapted to the models we're going to train:
    data as a flat matrix,
    labels as float 1-hot encodings.
'''

image_size = 28
num_labels = 10

def reformat(dataset, labels):
	dataset = dataset.reshape(-1, image_size*image_size).astype(np.float32)
	# map label 0 or 1 to vector
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


#build the computation graph
train_subset = 10000
graph = tf.Graph()
with graph.as_default():
	#define tensorflow constants
	tf_train_dataset = tf.constant(train_dataset[:train_subset,:])
	tf_train_labels = tf.constant(train_labels[:train_subset,:])
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	
	#define tensorflow variables, truncated_normal distribution
weights_2	weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))
	
	#define training computation
	logits = tf.matmul(tf_train_dataset, weights) + biases
	#cross_entropy between groud truth label and logistic predicts labels
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
	
	#define optimizer, using gradient descent
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	#define accuracy, not part of training, but let us watch the accuracy figures as we train
	#JST: compare the hypthesis between training, validation and test set
	#weights,biases are our training result
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights)+biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset,weights)+biases)

num_steps = 801
def accuracy(predictions, labels):
	#np.argmax(aa,1), return the index_array along the row axis of max value [0:colmun axis]
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels,1))/predictions.shape[0])

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run() # initial the constant and variables only once
	print('Initialized')
	for step in range(num_steps):
		#run optimizer,loss,train_prediction
		_,l,predictions = session.run([optimizer, loss, train_prediction])
		if (step %100 == 0):
			print('Loss at step %d: %f' % (step, l))
			print('Training accuracy:%.1f%%' % accuracy(predictions, train_labels[:train_subset,:]))
			#Calling .eval() on valid_prediction is basically like calling session.run(xxx), valid_prediction.eval() = session.run(valid_prediction)
			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
			print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
	session.close()	
print('Let us now switch to stochastic gradient descent training instead, which is much faster.')

batch_size = 128
sgd_graph = tf.Graph()
with sgd_graph.as_default():
	#now all the training data into placeholder  which will be fed actual data at every call of session.run()
	tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	
	weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))
	
	#define training computation
	logits = tf.matmul(tf_train_dataset, weights) + biases
	#cross_entropy between groud truth label and logistic predicts labels
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
	
	#define optimizer, using gradient descent
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	#define accuracy, not part of training, but let us watch the accuracy figures as we train
	#JST: compare the hypthesis between training, validation and test set
	#weights,biases are our training result
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights)+biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset,weights)+biases)

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
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = jst.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 500 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
			print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


