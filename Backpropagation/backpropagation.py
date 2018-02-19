import pandas as pd
from random import random
from random import randrange
from math import exp
from sklearn.metrics import confusion_matrix


'''
Preprocessing of the quality column. Anything less than or equal to 5
is changed to a 0 and anything greater becomes a 1.
'''
def preprocess(data):
	data.loc[data["quality"] <= 5, "quality"] = 0
	data.loc[data["quality"] > 5, "quality"] = 1
	return data


'''
Initialize the network by assigning the hidden layer and output layer
to random weights.
'''
def initialize_network(inputs, hidden, outputs):
	network = list()

	# create hidden layers with 1 more node than the length of the input layer
	hidden_layer = [{'weights':[random() for i in range(inputs + 1)]} for i in range(hidden)]
	network.append(hidden_layer)

	# create output layer with 1 more node than the length of the hidden layer
	output_layer = [{'weights':[random() for i in range(hidden + 1)]} for i in range(outputs)]
	network.append(output_layer)

	return network


'''
Calculate the activation of one neuron given an input.
'''
def activation(weights, inputs):

	# assume bias last weight is the bias
	activation = weights[-1]

	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation


'''
Once a neuron is activated, we calculate the output of that neuron
using the sigmoid function.
'''
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))


'''
Forward propagate through the layers until we have reached an output.
'''
def forward_propagate(network, row):
	inputs = row

	# loop through layers
	for layer in network:
		new_inputs = []

		# loop through nodes in each layer
		for neuron in layer:

			# calculate activation
			activate = activation(neuron["weights"], inputs)

			# feed activation to sigmoid function
			neuron["output"] = transfer(activate)
			new_inputs.append(neuron["output"])

		# the new outputs become the inputs for the next layer
		inputs = new_inputs

	return inputs



'''
Calculate the slope of an output node to be used in Backpropagation.
'''
def transfer_derivative(output):
	return output * (1.0 - output)


'''
Calculate errors of each neuron in every layer and save them as delta.
'''
def back_propagate_error(network, expected):

	# backpropagation loops backwards
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network) - 1:
			for j in range(len(layer)):
				error = 0.0

				# loop through output neurons
				for neuron in network[i+1]:

					# calculate error of each output neuron
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:

			# first initialize delta by calculating desired - output
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])

		# take all errors for each neuron
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])




'''
Update the weights based on the error rate calculated and saved into delta and through
the use of momentum.
'''
def update_weights(network, row, learning_rate, momentum=0.5):
	for i in range(len(network)):

		# all columns except last one which we are trying to predict
		inputs = row[:-1]

		# keeps track of momentum
		velocity = [0] * len(inputs)

		# update weights using the output from the previous layer
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
			velocity = [0] * len(inputs)

		# update weights
		for neuron in network[i]:
			for j in range(len(inputs)):

				# calculates the momentum by keeping track of the previous step in
				# the backpropagation
				velocity[j] = learning_rate * neuron['delta'] * inputs[j] + momentum * velocity[max(0, j-1)]
				neuron['weights'][j] += velocity[j]

			neuron['weights'][-1] += learning_rate * neuron['delta']



'''
Train the network by forward propagating and then updating the weights using
backpropagation which is dependent on the error rate.
'''
def train_network(network, train, learning_rate, epoch, outputs):
	for e in range(epoch):
		sum_error = 0
		for row in train:
			f_output = forward_propagate(network, row)

			# there are two output nodes for 1 and 0
			expected = [0 for i in range(outputs)]
			expected[int(row[-1])] = 1

			# sum sqaured error of desired - expected
			sum_error += sum([(expected[i]-f_output[i])**2 for i in range(len(expected))])

			# update weights through backpropagation
			back_propagate_error(network, expected)
			update_weights(network, row, learning_rate)

		print('>epoch=%d, lrate=%.3f, error=%.3f' % (e, learning_rate, sum_error))

	print('weights: ', network)


'''
Find max and min values of every column.
'''
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats


'''
Rescale dataset columns to the range 0-1.
'''
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):

			# normalize (row - min / max - min)
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


'''
Calculate accuracy percentage.
'''
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0



'''
Write list to a text file.
'''
def write_list(alist):
	result = open('result.txt', 'w')
	for p in alist:
		result.write("%s\n" % p)



'''
Evaluate the backpropagation algorithm and print out the accuracy metrics
as well as the confusion matrix of the results.
'''
def evaluate_algorithm_normal_split(dataset, algorithm, *args):
	
	# split 80/20
	split = round(len(dataset) * 0.2)
	train = dataset[:-split]
	test = dataset[-split:]

	scores = list()

	# get predictions for each row
	predicted = algorithm(train, test, *args)

	# write predictions to a text file
	write_list(predicted)
	
	# get the actual labels
	actual = [row[-1] for row in test]

	accuracy = accuracy_metric(actual, predicted)

	# display the confusion matrix
	p_actual = pd.Series(actual, name="Actual")
	p_predicted = pd.Series(predicted, name="Predicted")
	print(pd.crosstab(p_actual, p_predicted))

	scores.append(accuracy)

	return scores



'''
Cross-validation for testing the dataset on different sets of
testing data.
'''
def cross_validation_split(dataset, folds):
	dataset_split = list()
	dataset_copy = list(dataset)

	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		
		# while fold is not full
		while len(fold) < fold_size:
			
			# pop a random element and add it into the fold
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))

		# append the fold
		dataset_split.append(fold)

	return dataset_split


'''
Evaluates the backpropagation algorithm using a cross-validation split.
'''
def evaluate_algorithm_CV(dataset, algorithm, number_folds, *args):

	# returns the split test sets
	folds = cross_validation_split(dataset, number_folds)

	scores = list()

	# variables for confusion matrix
	true_positive = false_negative = false_positive = true_negative = 0

	for fold in folds:

		# one of the folds will be the test set
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()

		for row in fold:

			# all except the last column is used in the test set
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None

		# get predictions for each row
		predicted = algorithm(train_set, test_set, *args)

		# get the actual labels
		actual = [row[-1] for row in fold]

		accuracy = accuracy_metric(actual, predicted)

		# display the confusion matrix
		p_actual = pd.Series(actual, name="Actual")
		p_predicted = pd.Series(predicted, name="Predicted")
		cross_table = pd.crosstab(p_actual, p_predicted)
		print(cross_table)

		# sum all of the results because the average of these will be used
		# in the confusion matrix
		true_positive += cross_table[0][0]
		false_negative += cross_table[1][0]
		true_negative += cross_table[1][1]
		false_positive += cross_table[0][1]

		scores.append(accuracy)

	# average the results
	true_positive = int(true_positive / len(folds))
	false_negative = int(false_negative / len(folds))
	false_positive = int(false_positive / len(folds))
	true_negative = int(true_negative / len(folds))

	# create a new confusion matrix using the average of all the
	# results from the cross validation
	cross_table[0][0] = true_positive
	cross_table[1][0] = false_negative
	cross_table[1][1] = true_negative
	cross_table[0][1] = false_positive

	print(cross_table)

	return scores



'''
Make a prediction within the network.
'''
def predict(network, row):
	outputs = forward_propagate(network, row)

	# return the node with the highest score
	return outputs.index(max(outputs))


'''
Backpropagation algorithm.
'''
def back_propagation(train, test, learning_rate, epoch, hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, hidden, n_outputs)
	train_network(network, train, learning_rate, epoch, n_outputs)
	predictions = list()

	# make a prediction for every row in the dataset
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return predictions




def main():

	# preprocess
	df = pd.read_csv("wine.csv")
	df = preprocess(df)

	# convert series to list
	dataset = df.as_matrix().tolist()

	# normalize
	minmax = dataset_minmax(dataset)
	normalize_dataset(dataset, minmax)
	
	folds = 5
	learning_rate = 0.3
	epoch = 300
	hidden = 12

	# statistics
	scores = evaluate_algorithm_CV(dataset, back_propagation, folds, learning_rate, epoch, hidden)
	avg_score = sum(scores) / len(scores)
	print('Scores: %s' % scores)
	print('Average Score: %s' % avg_score)



main()








