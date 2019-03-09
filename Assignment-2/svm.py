import sys
import csv
import pandas as pd
import numpy as np
import cvxopt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from svmutil import *

# getting the training data
def get_train_params(train_data_path,issubset,digit):
	train_data = np.array(pd.read_csv(train_data_path,header=None,dtype=float).values)
	train_output = np.array(train_data[:,784:785])

	if issubset==True:
		print("treating " + str(digit) + " as class 1 and " + str(digit+1) + " as class -1")
		train_data = train_data[np.ix_((train_data[:,784]==digit) | (train_data[:,784]==digit+1))]
		train_output = 2*digit + 1 - 2*train_data[:,784:785]

	train_data = train_data/256
	return (np.asmatrix(train_data[:,0:784]),np.asmatrix(train_output))

# get the testing data
def get_test_params(test_data_path,issubset,digit):
	test_data = np.array(pd.read_csv(test_data_path,header=None,dtype=float).values)
	test_output = np.array(test_data[:,784:785])

	if issubset==True:
		test_data = test_data[np.ix_((test_data[:,784]==digit) | (test_data[:,784]==digit+1))]
		test_output = 2*digit + 1 - 2*test_data[:,784:785]
	
	test_data = test_data/256
	return (np.asmatrix(test_data[:,0:784]),np.asmatrix(test_output))

# linear kernel
def linear_kernel_cvxopt(train_data,train_output,penalty):
	m = len(train_data)
	X_Y = np.multiply(train_data,train_output)
	
	P = cvxopt.matrix(np.dot(X_Y,X_Y.transpose()))
	q = cvxopt.matrix(-1*np.ones((m,1)))
	A = cvxopt.matrix(train_output.transpose())
	b = cvxopt.matrix(0.0)

	tmp1 = -1*np.identity(m)
	tmp2 = np.identity(m)
	G = cvxopt.matrix(np.vstack((tmp1,tmp2)))
	tmp1 = np.zeros((m,1))
	tmp2 = penalty*np.ones((m,1))
	h = cvxopt.matrix(np.vstack((tmp1,tmp2)))
	solution = cvxopt.solvers.qp(P,q,G,h,A,b)
	return solution

# gaussian kernel
def gaussian_kernel_cvxopt(train_data,train_output,gamma,penalty):
	m = len(train_data)
	kernel = np.asmatrix(np.zeros((m,m),dtype=float))
	X_XT = np.dot(train_data,train_data.transpose())
	for i in range(m):
		for j in range(m):
			kernel[i,j] = float(X_XT[i,i] + X_XT[j,j] - 2*X_XT[i,j])
	kernel = np.exp(-1*gamma*kernel)

	P = cvxopt.matrix(np.multiply(kernel,np.dot(train_output,train_output.transpose())))
	q = cvxopt.matrix(-1*np.ones((m,1)))
	A = cvxopt.matrix(train_output.transpose())
	b = cvxopt.matrix(0.0)

	tmp1 = -1*np.identity(m)
	tmp2 = np.identity(m)
	G = cvxopt.matrix(np.vstack((tmp1,tmp2)))
	tmp1 = np.zeros((m,1))
	tmp2 = penalty*np.ones((m,1))
	h = cvxopt.matrix(np.vstack((tmp1,tmp2)))
	solution = cvxopt.solvers.qp(P,q,G,h,A,b)
	return solution

# calculating the weight matrix for linear kernel
def calculate_linear_svm_params(kernel_soln,train_data,train_output,tolerance):
	nSV = 0
	(m,n) = (train_data.shape[0],train_data.shape[1])
	raveled = np.ravel(kernel_soln['x'])
	langrangian_params = np.arange(len(raveled)) [raveled>tolerance]
	weight_matrix = np.asmatrix(np.zeros((1,n),dtype=float))
	
	for i in langrangian_params:
		for j in range(n):
			weight_matrix[0,j]+=(raveled[i]*train_data[i,j]*train_output[i,0])
		nSV+=1

	idx_used_for_b = langrangian_params[0]
	b = train_output[idx_used_for_b] - np.dot(train_data[idx_used_for_b,:],weight_matrix.transpose())[0,0]
	return (weight_matrix,b,nSV)

# predicting using parameters supplied on the supplied test_data
def linear_kernel_svm_prediction(weight_matrix,b,test_data):
	predicted = np.asmatrix(np.zeros((len(test_data),1),dtype=int))
	val = np.dot(test_data,weight_matrix.transpose()) + b
	predicted = 2*np.multiply((val>0),np.ones((len(test_data),1))) - 1
	return predicted

# calculating the weight matrix for gaussian kernel
def gaussian_endgame(kernel_soln,train_data,train_output,test_data,tolerance,gamma):
	(m,n) = (train_data.shape[0],train_data.shape[1])
	raveled = np.ravel(kernel_soln['x'])
	nSV = 0

	X_train = np.sum(np.multiply(train_data,train_data),axis=1)
	X_test = np.sum(np.multiply(test_data,test_data),axis=1)
	X_train_X_test = np.dot(train_data,test_data.transpose())

	alpha_x_label = np.asmatrix(np.zeros((len(raveled),1),dtype=float))
	for i in range(len(raveled)):
		if raveled[i]>tolerance:
			print("alpha " + str(i) + "->" + str(raveled[i]))
			alpha_x_label[i,0] = train_output[i,0]*raveled[i]*(raveled[i]>tolerance)
			nSV+=1
	
	langrangian_params = np.arange(len(raveled)) [raveled>tolerance]
	prediction = np.zeros((len(test_data),1),dtype=int)

	if len(langrangian_params)<=0:
		print("No support vectors found for tolerance value= " + str(tolerance))
	else:
		idx_used_for_b = langrangian_params[0]
		b = np.sum(train_output - np.sum(np.multiply(alpha_x_label,np.exp(-1*gamma*np.sum(np.multiply(train_data-train_data[idx_used_for_b,:],train_data-train_data[idx_used_for_b,:]),axis=1)))))/(float(len(train_data)))
		print(str(b) + " is the value of b")
		# b = 0.14073416301529917
		for i in range(len(test_data)):
			prediction[i] = np.sign(np.sum(np.multiply(alpha_x_label,np.exp(-1*gamma*(X_train - 2*X_train_X_test[:,i] + X_test[i,0])))) + b)

	return (prediction,nSV)

# libsvm package
def libsvm(train_data,train_output,test_data,test_output,gamma,penalty):
	train_labels = []
	train_input = train_data.tolist()
	for j in range(train_output.shape[0]):
		train_labels.append(train_output[j,0])
	
	test_labels = []
	test_input = test_data.tolist()
	for j in range(test_output.shape[0]):
		test_labels.append(test_output[j,0])
	problem = svm_problem(train_labels,train_input)
	
	linear_param = svm_parameter("-s 0 -c 1 -t 0")
	linear_model = svm_train(problem,linear_param)
	linear_pred_lbl, linear_pred_acc, linear_pred_val = svm_predict(test_labels,test_input,linear_model)

	gaussian_param = svm_parameter("-s 0 -c " + str(penalty) + " -t 2 -g " + str(gamma))
	gaussian_model = svm_train(problem,gaussian_param)
	gaussian_pred_lbl, gaussian_pred_acc, gaussian_pred_val = svm_predict(test_labels,test_input,gaussian_model)

# main function
def main():
	# taking parameters from command line
	train_data_path = sys.argv[1]
	test_data_path = sys.argv[2]
	classification = sys.argv[3]
	part = sys.argv[4]

	# reading train and test data as array
	issubset = (classification=='0')
	print("reading data with normalization")
	digit = 5
	(train_data,train_output) = get_train_params(train_data_path,issubset,digit)
	(test_data,test_output) = get_test_params(test_data_path,issubset,digit)
	
	if issubset==True:
		if part == 'a':
			tolerance = 1e-4
			penalty = 1
			print("tolerance value for linear kernel = " + str(tolerance))
			linear_kernel_soln = linear_kernel_cvxopt(train_data,train_output,penalty)
			(weight_matrix,b,nSV) = calculate_linear_svm_params(linear_kernel_soln,train_data,train_output,tolerance)
			print(str(nSV) + " support vectors")
			predicted = linear_kernel_svm_prediction(weight_matrix,b,test_data)
			confatrix = confusion_matrix(test_output,predicted)
			print(confatrix)
		
		elif part =='b':
			gamma = 0.05
			penalty = 1
			tolerance = 1e-4
			print("tolerance value for gaussian kernel = " + str(tolerance))
			gaussian_kernel_soln = gaussian_kernel_cvxopt(train_data,train_output,gamma,penalty)
			(predicted,nSV) = gaussian_endgame(gaussian_kernel_soln,train_data,train_output,test_data,tolerance,gamma)
			print(str(nSV) + " support vectors")
			confatrix = confusion_matrix(test_output,predicted)
			print(confatrix)
		
		elif part == 'c':
			gamma = 0.05
			penalty = 1
			libsvm(train_data,train_output,test_data,test_output,gamma,penalty)

		else:
			print("No such part for this classification")

	else:
		if part == 'a':
			
		elif part =='b':
			
		elif part == 'c':
			
		elif part == 'd':

		else:
			print("No such part for this classification")
	
	return

if __name__ == "__main__":
	main()