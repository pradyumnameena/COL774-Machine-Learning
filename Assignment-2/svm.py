import csv
import pandas as pd
import numpy as np
import cvxopt
import cvxopt.solvers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# getting the training data
def get_train_params(issubset):
	train_data = pd.read_csv('ass2_data_svm/train.csv',header=None,dtype=float)
	if issubset==True:
		train_data = train_data.loc[(train_data[784] == 5) | (train_data[784] == 6)].values
	train_output = train_data[:,784:785]
	train_data = train_data[:,0:784]/256
	for i in range(len(train_output)):
		if train_output[i,0] == 5:
			train_output[i,0] = 1
		else:
			train_output[i,0] = -1
	return (train_data,train_output)

# get the testing data
def get_test_params(issubset):
	test_data = pd.read_csv('ass2_data_svm/test.csv',header=None,dtype=float)
	if issubset==True:
		test_data = test_data.loc[(test_data[784] == 5) | (test_data[784] == 6)].values
	test_output = test_data[:,784:785]
	test_data = test_data[:,0:784]/256
	for i in range(len(test_output)):
		if test_output[i,0] == 5:
			test_output[i,0] = 1
		else:
			test_output[i,0] = -1
	return (test_data,test_output)

# linear kernel
def linear_kernel(train_data,train_output):
	# for cvxopt use
	m = len(train_data)
	X_Y = np.multiply(train_data,train_output)
	
	P = cvxopt.matrix(np.dot(X_Y,X_Y.transpose()))
	q = cvxopt.matrix(-2*np.ones((m,1)))
	A = cvxopt.matrix(train_output.transpose())
	b = cvxopt.matrix(0.0)

	tmp1 = -1*np.identity(m)
	tmp2 = np.identity(m)
	G = cvxopt.matrix(np.vstack((tmp1,tmp2)))
	tmp1 = np.zeros(m)
	tmp2 = np.ones(m)
	h = cvxopt.matrix(np.hstack((tmp1,tmp2)))
	solution = cvxopt.solvers.qp(P,q,G,h,A,b)
	return solution

# gaussian function calculator
def gaussain_func(a,b,gamma):
	diff = a-b
	rv = np.exp(-1*gamma*float(np.sum(diff*diff)))
	return rv

# gaussian kernel
def gaussian_kernel(train_data,train_output,gamma):
	# for cvxopt use
	m = len(train_data)
	n = train_data.shape[1]
	X_Y = np.asmatrix(np.zeros((m,n),dtype=float))
	for i in range(m):
		for j in range(n):
			X_Y[i,j] = gaussain_func(train_data[i,:],train_data[j,:],gamma)*train_output[i]
			# X_Y[i,j] = gaussain_func(train_data[i,:],train_data[j,:],gamma)*train_output[i]*train_output[j]
	
	P = cvxopt.matrix(np.dot(X_Y,X_Y.transpose()))
	q = cvxopt.matrix(-2*np.ones((m,1)))
	A = cvxopt.matrix(train_output.transpose())
	b = cvxopt.matrix(0.0)

	tmp1 = -1*np.identity(m)
	tmp2 = np.identity(m)
	G = cvxopt.matrix(np.vstack((tmp1,tmp2)))
	tmp1 = np.zeros(m)
	tmp2 = np.ones(m)
	h = cvxopt.matrix(np.hstack((tmp1,tmp2)))
	solution = cvxopt.solvers.qp(P,q,G,h,A,b)
	return solution

# calculating the weight matrix
def calculate_svm_params(kernel_soln,train_data,train_output,tolerance):
	(m,n) = (train_data.shape[0],train_data.shape[1])
	raveled = np.ravel(kernel_soln['x'])
	langrangian_params = np.arange(len(raveled)) [raveled>tolerance]
	weight_matrix = np.asmatrix(np.zeros((1,n),dtype=float))
	
	for i in langrangian_params:
		for j in range(n):
			weight_matrix[0,j]+=(raveled[i]*train_data[i,j]*train_output[i,0])
	
	idx_used_for_b = langrangian_params[0]
	b = train_output[idx_used_for_b] - np.dot(train_data[idx_used_for_b,:],weight_matrix.transpose())[0,0]
	
	return (weight_matrix,b)

# predicting using parameters supplied on the supplied test_data
def svm_prediction(weight_matrix,b,test_data):
	predicted = np.asmatrix(np.zeros((len(test_data),1),dtype=int))
	for i in range(len(test_data)):
		val = np.dot(test_data[i,:],weight_matrix.transpose())[0,0] + b
		if val>0:
			predicted[i] = 1
		else:
			predicted[i] = -1
	return predicted

# main function
def main():
	print("treating 5 as class 1 and 6 as class -1")
	
	(train_data,train_output) = get_train_params(True)
	print("normalized training data retrieved")

	(test_data,test_output) = get_test_params(True)
	print("normalized test data retrieved")

	linear_kernel_soln = linear_kernel(train_data,train_output)
	print("linear kernel solution found")
	
	tolerance = 1e-8
	(weight_matrix,b) = calculate_svm_params(linear_kernel_soln,train_data,train_output,tolerance)
	print("svm parameters computed")

	predicted = svm_prediction(weight_matrix,b,test_data)
	print("prediction complete")

	confatrix = confusion_matrix(test_output,predicted)
	print("confusion matrix computed")
	
	print("below is confusion matrix for linear kernel")
	print(confatrix)

	# below is for gaussian kernel
	gamma = 0.05
	gaussian_kernel_soln = gaussian_kernel(train_data,train_output,gamma)
	print("gaussian kernel solution found")
	
	tolerance = 1e-8
	(weight_matrix,b) = calculate_svm_params(gaussian_kernel_soln,train_data,train_output,tolerance)
	print("svm parameters computed")

	predicted = svm_prediction(weight_matrix,b,test_data)
	print("prediction complete")

	confatrix = confusion_matrix(test_output,predicted)
	print("confusion matrix computed")
	
	print("below is confusion matrix for gaussian kernel")
	print(confatrix)
	return

if __name__ == "__main__":
	main()