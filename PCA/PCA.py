import numpy as np
import pandas as pd
import random
import csv


'''
Statistical implementation for calculating the first
principal component.
'''
def PCA_stat(df):

	# convert dataframe to a numpy matrix
	sound = df.as_matrix()
	
	# calculate the mean of each column
	mean = np.mean(sound.T, axis=1)
	print(mean)

	# substract the mean from each column
	C = sound - mean
	print(C)

	# calculate the covariance matrix
	cov = covariance(C)
	print(cov)

	# get the eigen vectors and eigen values
	values, vectors = np.linalg.eig(cov)
	print(vectors)
	print(values)

	# project the data
	P = vectors.T.dot(C.T)
	print(P.T)


'''
A manual implementation for calculating the covariance matrix
of any matrix.
'''
def covariance(df):
	fact = len(df) - 1
	cov = np.dot(df.T, df.conj()) / fact
	return cov


'''
PCA network implementation. Using ANN techniques to calculate
the first principal component.
'''
def PCA_NN(df):

	# convert to numpy matrix
	sound = df.as_matrix()

	# intialize weights to some random values
	weights = random.sample(range(1, 100), 2)
	weights = [x/100 for x in weights]
	
	learning_rate = 1

	# loop through rows
	for x in sound:

		# dot product of weights and row
		y = np.dot(weights, x)
		
		# scalar multiplication
		W_1 = y * x
		
		# weight constraint
		K = y*y

		# put together the Hebbian learning rule
		W_2 = np.multiply(K, weights)
		W = np.subtract(W_1,  W_2)
		
		# update weights
		weights += learning_rate * W
	
	# apply the dot product on every row with the new weight and save the new output
	new_output = []
	for x in sound:
		output = np.dot(weights, x)
		new_output.append(output)

	# save the output to a csv
	new_output = np.array(new_output)
	np.savetxt("output.csv", new_output, delimiter=",")	



def main():

	df = pd.read_csv("sound.csv")
	PCA_stat(df)
	PCA_NN(df)

main()