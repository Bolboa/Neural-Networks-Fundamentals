
from random import choice
import numpy as np
import sklearn.linear_model as ske
from sklearn import datasets, svm, model_selection, tree, metrics, preprocessing


def perceptron(training_data, test):

	# function for checking which class the flower belongs to
	unit_step = lambda x: 0 if x < 0 else 1

	# random weights
	w = np.random.rand(5)
	errors = []
	
	# learning rate
	learning_rate = 0.2

	# number of runs
	epoch = 100


	for i in range(epoch):
		
		# choose random order every epoch
		X, Y = choice(training_data)
		
		# find the dot product
		result = np.dot(w, X)

		# calculate the error
		error = Y - unit_step(result)
		errors.append(error)

		# minimize the error by changing the weight
		w += learning_rate * error * X

	output = []

	# predict the test data
	for X in test:
		result = np.dot(X, w)
		output.append(result)
		print("{}: {} -> {}".format(X[:4], result, unit_step(result)))

	# calculate the total squared error
	total_squared = 0
	for err in errors:
		total_squared += err^2

	return output


def generate_three(expected, target):

	flower = []

	# we generate three different data sets to be used by three different perceptrons
	# [Iris-virginica, Iris-setosa + Iris-versicolor], [Iris-setosa, Iris-virginica + Iris-versicolor],
	# [Iris-versicolor, Iris-setosa + Iris-virginica]
	if target == 1:
		for i in range(len(expected)):
			if expected[i] == 'Iris-setosa' or expected[i] == 'Iris-versicolor':
				flower.append(0)
			else:
				flower.append(1)	
	elif target == 2:
		for i in range(len(expected)):
			if expected[i] == 'Iris-virginica' or expected[i] == 'Iris-versicolor':
				flower.append(0)
			else:
				flower.append(1)
	else:
		for i in range(len(expected)):
			if expected[i] == 'Iris-setosa' or expected[i] == 'Iris-virginica':
				flower.append(0)
			else:
				flower.append(1)

	return flower



def multi_prediction(pred_vg, pred_s, pred_vs):

	new_output = []

	# this is the one vs all approach, we take
	# the best prediction
	for i in range(len(pred_vg)):
		if pred_vg > pred_vs and pred_vg > pred_s:
			new_output.append(0)
		elif pred_vs > pred_vg and pred_vs > pred_s:
			new_output.append(1)
		else:
			new_output.append(2)
	return new_output



def perceptron_model(training_data, expected):

	expected = np.array(expected)

	clf_perceptron = ske.Perceptron(
		penalty=None, alpha=0.0001, fit_intercept=True, max_iter=None, tol=None, 
		shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, 
		class_weight=None, warm_start=False, n_iter=None)

	clf_perceptron.fit(training_data, expected)
	print('accuracy:', clf_perceptron.score(training_data, expected))
	return clf_perceptron


def calculate_accuracy(actual, predicted):
	
	correct = 0
	incorrect = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
		else:
			incorrect += 1

	print(correct, incorrect)


def main():
	
	# read train and test
	gen = np.genfromtxt("train.txt", dtype=None)
	gen_test = np.genfromtxt("test.txt", dtype=None)
	
	# formatting
	train = []
	for g in gen:
		train.append(g.decode('UTF-8'))

	test = []
	for g in gen_test:
		test.append(g.decode('UTF-8'))

	expected = []
	for i in range(len(train)):
		t_split = train[i].split(",")
		expected.append(t_split[4])
		t_split[4] = 1
		train[i] = np.array(t_split).astype(float)
	
	check_test = []
	for i in range(len(test)):
		t_split = test[i].split(",")
		check_test.append(t_split[4])
		t_split[4] = 1
		test[i] = np.array(t_split).astype(float)

	# generate the three data sets
	expected1 = generate_three(expected, 1)
	expected2 = generate_three(expected, 2)
	expected3 = generate_three(expected, 3)

	# generate the three trainin datas
	training_data_vg = []
	for i in range(len(train)):
		training_data_vg.append((train[i], expected1[i]))

	training_data_s = []
	for i in range(len(train)):
		training_data_s.append((train[i], expected2[i]))

	training_data_vs = []
	for i in range(len(train)):
		training_data_vs.append((train[i], expected3[i]))

	# use the perceptron for all three training datas
	pred_vg = perceptron(training_data_vg, test)
	pred_s = perceptron(training_data_s, test)
	pred_vs = perceptron(training_data_vs, test)

	# one vs all
	final = multi_prediction(pred_vg, pred_s, pred_vs)

	output_file = open('output.txt', 'w')
	actual = ''
	predicted = ''

	for i in range(len(final)):
		
		if final[i] == 0:
			predicted = 'Iris-virginica'
		elif final[i] == 1:
			predicted = 'Iris-setosa'
		elif final[i] == 2:
			predicted = 'Iris-versicolor'
		actual = check_test[i]

		output_file.write('{0} {1} \n'.format(predicted, actual))


	# the following is the perceptron tool provided by scikit-learn
	clf_perceptron = perceptron_model(train, expected)
	

	output_file2 = open('output_model.txt', 'w')

	predictions = clf_perceptron.predict(test)	

	for i in range(len(predictions)):
		output_file2.write('{0} {1} \n'.format(predictions[i], check_test[i]))

main()