import sys
import csv
import pandas as pd
import numpy as np
import cvxopt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from svmutil import *

# getting the training data
def get_train_params(train_data_path,issubset,digit1,digit2):
	train_data = np.array(pd.read_csv(train_data_path,header=None,dtype=float).values)
	train_output = np.array(train_data[:,784:785])

	if issubset==True:
		train_data = train_data[np.ix_((train_data[:,784]==digit1) | (train_data[:,784]==digit2))]
		train_output = train_data[:,784:785]
		
		for i in range(len(train_data)):
			if train_output[i,0] == digit1:
				train_output[i,0] = 1
			else:
				train_output[i,0] = -1

	train_data = train_data/256
	return (np.asmatrix(train_data[:,0:784]),np.asmatrix(train_output))

# get the testing data
def get_test_params(test_data_path,issubset,digit1,digit2):
	test_data = np.array(pd.read_csv(test_data_path,header=None,dtype=float).values)
	test_output = np.array(test_data[:,784:785])

	if issubset==True:
		test_data = test_data[np.ix_((test_data[:,784]==digit1) | (test_data[:,784]==digit2))]
		test_output = np.array(test_data[:,784:785])
		for i in range(len(test_data)):
			if test_output[i,0] == digit1:
				test_output[i,0] = 1
			else:
				test_output[i,0] = -1
	
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
	b = 0
	if nSV==0:
		print("No support vectors found for tolerance value of " + str(tolerance))
	else:
		for sv_idx in langrangian_params:
			b+=(train_output[sv_idx,0] - np.dot(train_data[sv_idx,:],weight_matrix.transpose())[0,0])
		b = b/(float(len(langrangian_params)))
		print(str(b) + " is the value of b")
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
			alpha_x_label[i,0] = train_output[i,0]*raveled[i]*(raveled[i]>tolerance)
			nSV+=1
	
	langrangian_params = np.arange(len(raveled)) [raveled>tolerance]
	prediction = np.zeros((len(test_data),1),dtype=int)

	if len(langrangian_params)<=0:
		print("No support vectors found for tolerance value= " + str(tolerance))
	else:
		b = 0
		for sv_idx in langrangian_params:
			b+=(train_output[sv_idx,0] - np.sum(np.multiply(alpha_x_label,np.exp(-1*gamma*np.sum(np.multiply(train_data-train_data[sv_idx,:],train_data-train_data[sv_idx,:]),axis=1)))))
		b = b/(float(len(langrangian_params)))
		print(str(b) + " is the value of b")
		
		for i in range(len(test_data)):
			prediction[i] = np.sign(np.sum(np.multiply(alpha_x_label,np.exp(-1*gamma*(X_train - 2*X_train_X_test[:,i] + X_test[i,0])))) + b)

	return (prediction,nSV)

# for computing predictions for multiclass classification
def gaussian_infinity_wars(kernel_soln_x,train_data,train_output,test_data,tolerance,gamma):
	prediction = np.asmatrix(np.ones((len(test_data),1),dtype=int))
	raveled = np.asmatrix(kernel_soln_x)
	
	X_train = np.sum(np.multiply(train_data,train_data),axis=1)
	X_test = np.sum(np.multiply(test_data,test_data),axis=1)
	X_train_X_test = np.dot(train_data,test_data.transpose())

	alpha_x_label = np.multiply(train_output,np.multiply(raveled,raveled>tolerance))
	langrangian_params = np.nonzero(raveled>tolerance)[0]

	if len(langrangian_params)==0:
		print("No support vectors found for tolerance value= " + str(tolerance))
	else:
		b = 0
		for sv_idx in langrangian_params:
			b+=(train_output[sv_idx,0] - np.sum(np.multiply(alpha_x_label,np.exp(-1*gamma*np.sum(np.multiply(train_data-train_data[sv_idx,:],train_data-train_data[sv_idx,:]),axis=1)))))
		b = b/(float(len(langrangian_params)))
		
		for i in range(len(test_data)):
			prediction[i,0] = np.sign(np.sum(np.multiply(alpha_x_label,np.exp(-1*gamma*(X_train - 2*X_train_X_test[:,i] + X_test[i,0])))) + b)

	return prediction

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

# multiclass gaussian
def multiclass_svm_cvxopt(train_data_path,test_data_path,gamma,penalty,tolerance):
	svm_dict = {}
	num_max = 2
	# learning parameters phase
	for i in range(1+num_max):
		for j in range(i):
			idx = str(i)+str(j)
			svm_dict[idx] = []
			(train_data,train_output) = get_train_params(train_data_path,True,i,j)
			kernel_soln = gaussian_kernel_cvxopt(train_data,train_output,gamma,penalty)
			svm_dict[idx] = np.ravel(kernel_soln['x']).tolist()
			print("langrangian parameters for svm with index value " + idx + " computed")

	# prediction phase
	(test_data,test_output) = get_test_params(test_data_path,False,0,0)
	prediction_dict = {}
	for i in range(len(test_data)):
		prediction_dict[i] = [0,0,0,0,0,0,0,0,0,0]
	prediction = np.asmatrix(np.zeros((len(test_data),1),dtype=int))
	
	for i in range(1+num_max):
		for j in range(i):
			idx = str(i)+str(j)
			kernel_soln_x = svm_dict[idx]
			(train_data,train_output) = get_train_params(train_data_path,True,i,j)
			svm_prediction = gaussian_infinity_wars(kernel_soln_x,train_data,train_output,test_data,tolerance,gamma)
			
			for k in range(len(svm_prediction)):
				if svm_prediction[k,0] == 1:
					prediction_dict[k][i]+=1
				else:
					prediction_dict[k][j]+=1
			print("predictions for svm with index value " + idx + " done")

	for i in range(len(test_data)):
		prediction[i] = np.argmax(prediction_dict[i])
	return (test_output,np.array(prediction))

# multiclass gaussian libsvm
def multiclass_svm_libsvm(train_data_path,test_data_path,gamma,penalty):
	svm_dict = {}
	num_max = 2
	# learning parameters phase
	for i in range(1+num_max):
		for j in range(i):
			idx = str(i)+str(j)
			svm_dict[idx] = []
			(train_data,train_output) = get_train_params(train_data_path,True,i,j)
			kernel_soln = gaussian_kernel_cvxopt(train_data,train_output,gamma,penalty)
			svm_dict[idx] = np.ravel(kernel_soln['x']).tolist()
			print("langrangian parameters for svm with index value " + idx + " computed")

	# prediction phase
	(test_data,test_output) = get_test_params(test_data_path,False,0,0)
	prediction_dict = {}
	for i in range(len(test_data)):
		prediction_dict[i] = [0,0,0,0,0,0,0,0,0,0]
	prediction = np.asmatrix(np.zeros((len(test_data),1),dtype=int))
	
	for i in range(1+num_max):
		for j in range(i):
			idx = str(i)+str(j)
			kernel_soln_x = svm_dict[idx]
			(train_data,train_output) = get_train_params(train_data_path,True,i,j)
			svm_prediction = gaussian_infinity_wars(kernel_soln_x,train_data,train_output,test_data,tolerance,gamma)
			
			for k in range(len(svm_prediction)):
				if svm_prediction[k,0] == 1:
					prediction_dict[k][i]+=1
				else:
					prediction_dict[k][j]+=1
			print("predictions for svm with index value " + idx + " done")

	for i in range(len(test_data)):
		prediction[i] = np.argmax(prediction_dict[i])
	return (test_output,np.array(prediction))

# main function
def main():
	# taking parameters from command line
	train_data_path = sys.argv[1]
	test_data_path = sys.argv[2]
	classification = sys.argv[3]
	part = sys.argv[4]
	issubset = (classification=='0')

	if issubset==True:
		
		# reading train and test data as array
		print("reading data with normalization")
		digit1 = 5
		digit2 = 6
		(train_data,train_output) = get_train_params(train_data_path,issubset,digit1,digit2)
		(test_data,test_output) = get_test_params(test_data_path,issubset,digit1,digit2)

		if part == 'a':
			tolerance = 1e-4
			penalty = 1
			print("tolerance value for linear kernel for binary classification = " + str(tolerance))
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
			print("tolerance value for gaussian kernel for binary classification = " + str(tolerance))
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
			print("No such part for binary classification")

	else:

		if part == 'a':
			gamma = 0.05
			penalty = 1
			tolerance = 1e-6
			print("tolerance value for gaussian kernel for multiclass classification= " + str(tolerance))
			(test_output,prediction) = multiclass_svm_cvxopt(train_data_path,test_data_path,gamma,penalty,tolerance)
			confatrix = confusion_matrix(test_output,prediction)
			print(confatrix)

		elif part =='b':
			gamma = 0.05
			penalty = 1
			(test_output,prediction) = multiclass_svm_libsvm(train_data_path,test_data_path,gamma,penalty)
			confatrix = confusion_matrix(test_output,prediction)
			print(confatrix)

		elif part == 'd':
			print("No such part for multiclass classification")
			
		else:
			print("No such part for multiclass classification")
	
	return

if __name__ == "__main__":
	main()