import sys
import csv
import time
import cvxopt
import numpy as np
import pandas as pd
from svmutil import *
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def get_data(data_path,issubset,digit1,digit2):
	train_data = np.array(pd.read_csv(data_path,header=None,dtype=float).values)
	train_output = np.array(train_data[:,784:785])

	if issubset==True:
		train_data = train_data[np.ix_((train_data[:,784]==digit1) | (train_data[:,784]==digit2))]
		train_output = train_data[:,784:785]
		for i in range(len(train_data)):
			if train_output[i,0] == digit1:
				train_output[i,0] = 1
			else:
				train_output[i,0] = -1

	train_data = train_data/255
	return (np.asmatrix(train_data[:,0:784]),np.asmatrix(train_output))

def draw_confusion(confatrix):
	plt.imshow(confatrix)
	plt.title("Confusion Matrix")
	plt.colorbar()
	plt.set_cmap("Greens")
	plt.ylabel("True labels")
	plt.xlabel("Predicted label")
	plt.show()

# Linear kernel using cvxopt for binary classification
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
	
	# writing indices of support vectors into text file 
	print("Indices of support vectors have been stored in linear_support_vector_indices.txt")
	np.savetxt("linear_support_vector_indices.txt", langrangian_params , delimiter=', ',fmt='%d')

	# writing weight matrix into text file
	print("Weight matrix has been stored in weight_matrix.txt")
	with open('weight_matrix.txt','a') as f:
		for line in weight_matrix:
			np.savetxt(f, line, fmt='%.2f')

	b = 0
	if nSV==0:
		print("No support vectors found for tolerance value of " + str(tolerance))
	else:
		for sv_idx in langrangian_params:
			b+=(train_output[sv_idx,0] - np.dot(train_data[sv_idx,:],weight_matrix.transpose())[0,0])
		b = b/(float(len(langrangian_params)))
		print(str(b) + " is the value of b")
	return (weight_matrix,b,nSV)

def linear_kernel_svm_prediction(weight_matrix,b,test_data):
	predicted = np.asmatrix(np.zeros((len(test_data),1),dtype=int))
	val = np.dot(test_data,weight_matrix.transpose()) + b
	predicted = 2*np.multiply((val>0),np.ones((len(test_data),1))) - 1
	return predicted

# Gaussian kernel using cvxopt for binary classification
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

def gaussian_prediction_cvxopt(kernel_soln,train_data,train_output,test_data,tolerance,gamma):
	(m,n) = (train_data.shape[0],train_data.shape[1])
	raveled = np.ravel(kernel_soln['x'])
	nSV = 0

	X_train = np.sum(np.multiply(train_data,train_data),axis=1)
	X_test = np.sum(np.multiply(test_data,test_data),axis=1)
	X_train_X_test = np.dot(train_data,test_data.transpose())

	alpha_x_label = np.asmatrix(np.zeros((len(raveled),1),dtype=float))
	for i in range(len(raveled)):
		if raveled[i]>tolerance:
			alpha_x_label[i,0] = train_output[i,0]*raveled[i]
			nSV+=1
	
	langrangian_params = np.arange(len(raveled)) [raveled>tolerance]
	prediction = np.zeros((len(test_data),1),dtype=int)

	# writing indices of support vectors into text file 
	print("Indices of support vectors have been saved in gaussian_support_vector_indices.txt")
	np.savetxt("gaussian_support_vector_indices.txt", langrangian_params , delimiter=', ',fmt='%d')

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

# Gaussian and linear kernel both using libsvm
def libsvm_both(train_data,train_output,test_data,test_output,gamma,penalty):
	# t_start = time.clock()
	train_labels = []
	train_input = train_data.tolist()
	for j in range(train_output.shape[0]):
		train_labels.append(train_output[j,0])
	
	test_labels = []
	test_input = test_data.tolist()
	for j in range(test_output.shape[0]):
		test_labels.append(test_output[j,0])
	
	# time_stamp1 = time.clock()
	# print(str(time_stamp1-t_start) + " is time taken for generating data in appropriate form")

	problem = svm_problem(train_labels,train_input)
	linear_param = svm_parameter("-s 0 -c 1 -t 0")
	linear_model = svm_train(problem,linear_param)
	# t_linear_end = time.clock()
	# print(str(t_linear_end-time_stamp1) + " for training linear kernel")
	linear_pred_lbl, linear_pred_acc, linear_pred_val = svm_predict(test_labels,test_input,linear_model)

	gaussian_param = svm_parameter("-s 0 -c " + str(penalty) + " -t 2 -g " + str(gamma))
	gaussian_model = svm_train(problem,gaussian_param)
	# t_gaussian_end = time.clock()
	# print(str(t_gaussian_end-t_linear_end) + " for training gaussian kernel")
	gaussian_pred_lbl, gaussian_pred_acc, gaussian_pred_val = svm_predict(test_labels,test_input,gaussian_model)

# multiclass classification using cvxopt
def gaussian_prediction_with_alphas(kernel_soln_x,train_data,train_output,test_data,tolerance,gamma):
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

def multiclass_svm_cvxopt(train_data_path,test_data_path,gamma,penalty,tolerance):
	svm_dict = {}
	num_max = 2
	# learning parameters phase
	for i in range(1+num_max):
		for j in range(i):
			idx = str(i)+str(j)
			svm_dict[idx] = []
			(train_data,train_output) = get_data(train_data_path,True,i,j)
			kernel_soln = gaussian_kernel_cvxopt(train_data,train_output,gamma,penalty)
			svm_dict[idx] = np.ravel(kernel_soln['x']).tolist()
			print("langrangian parameters for svm with index value " + idx + " computed")

	# prediction phase
	(test_data,test_output) = get_data(test_data_path,False,0,0)
	prediction_dict = {}
	for i in range(len(test_data)):
		prediction_dict[i] = [0,0,0,0,0,0,0,0,0,0]
	prediction = np.asmatrix(np.zeros((len(test_data),1),dtype=int))
	
	for i in range(1+num_max):
		for j in range(i):
			idx = str(i)+str(j)
			kernel_soln_x = svm_dict[idx]
			(train_data,train_output) = get_data(train_data_path,True,i,j)
			svm_prediction = gaussian_prediction_with_alphas(kernel_soln_x,train_data,train_output,test_data,tolerance,gamma)
			
			for k in range(len(svm_prediction)):
				if svm_prediction[k,0] == 1:
					prediction_dict[k][i]+=1
				else:
					prediction_dict[k][j]+=1
			print("predictions for svm with index value " + idx + " done")

	for i in range(len(test_data)):
		prediction[i] = np.argmax(prediction_dict[i])
	return (test_output,np.array(prediction))

# multiclass classification using libsvm
def multiclass_svm_libsvm_45(train_data_path,test_data_path,gamma,penalty):
	svm_dict = {}
	prediction_dict = {}
	num_max = 9
	(test_data,test_output) = get_data(test_data_path,False,0,0)
	for i in range(len(test_data)):
		prediction_dict[i] = [0,0,0,0,0,0,0,0,0,0]
	prediction = np.asmatrix(np.zeros((len(test_data),1),dtype=int))

	# learning parameters phase (45 individual svms)
	for i in range(1+num_max):
		for j in range(i):
			(train_data,train_output) = get_data(train_data_path,True,i,j)
			idx = str(i)+str(j)
			train_labels = []
			train_input = train_data.tolist()
			for i1 in range(train_output.shape[0]):
				train_labels.append(train_output[i1,0])
			
			test_labels = []
			test_input = test_data.tolist()
			for j1 in range(test_output.shape[0]):
				test_labels.append(test_output[j1,0])
			
			problem = svm_problem(train_labels,train_input)
			gaussian_param = svm_parameter("-s 0 -c " + str(penalty) + " -t 2 -g " + str(gamma))
			gaussian_model = svm_train(problem,gaussian_param)
			svm_prediction_lbl,svm_prediction_acc,svm_prediction_val = svm_predict(test_labels,test_input,gaussian_model)
			
			for k in range(len(svm_prediction_lbl)):
				if svm_prediction_lbl[k] == 1:
					prediction_dict[k][i]+=1
				else:
					prediction_dict[k][j]+=1
			print("prediction using gaussian kernel in libsvm completed for " + idx)

	for i in range(len(test_data)):
		prediction[i] = np.argmax(prediction_dict[i])
	
	return(test_output,prediction)

def multiclass_svm_libsvm(train_data_path,test_data_path,gamma,penalty):
	# libsvm_training_start = time.clock()
	
	(train_data,train_output) = get_data(train_data_path,False,0,0)
	(test_data,test_output) = get_data(test_data_path,False,0,0)

	train_labels = []
	train_input = train_data.tolist()
	for i1 in range(train_output.shape[0]):
		train_labels.append(train_output[i1,0])
	
	test_labels = []
	test_input = test_data.tolist()
	for j1 in range(test_output.shape[0]):
		test_labels.append(test_output[j1,0])

	problem = svm_problem(train_labels,train_input)
	gaussian_param = svm_parameter("-s 0 -c " + str(penalty) + " -t 2 -g " + str(gamma))
	gaussian_model = svm_train(problem,gaussian_param)
	
	libsvm_training_end = time.clock()
	# print(str(libsvm_training_end - libsvm_training_start) + " is the training time for multiclass libsvm")
	
	svm_prediction_lbl,svm_prediction_acc,svm_prediction_val = svm_predict(test_labels,test_input,gaussian_model)
	# print(svm_prediction_acc)
	return (test_output,svm_prediction_lbl)

def main():
	time_init = time.clock()
	train_data_path = sys.argv[1]
	test_data_path = sys.argv[2]
	classification = sys.argv[3]
	part = sys.argv[4]
	issubset = (classification=='0')
	if issubset==True:
		digit1 = 5
		digit2 = 6
		(train_data,train_output) = get_data(train_data_path,issubset,digit1,digit2)
		(test_data,test_output) = get_data(test_data_path,issubset,digit1,digit2)

		if part == 'a':
			tolerance = 1e-4
			penalty = 1
			print("tolerance,penalty for linear kernel for binary classification = " + str(tolerance) + "," + str(penalty))
			linear_kernel_soln = linear_kernel_cvxopt(train_data,train_output,penalty)
			(weight_matrix,b,nSV) = calculate_linear_svm_params(linear_kernel_soln,train_data,train_output,tolerance)
			print(str(nSV) + " support vectors")
			predicted = linear_kernel_svm_prediction(weight_matrix,b,test_data)
			confatrix = confusion_matrix(test_output,predicted)
			print("Confusion Matrix")
			print(confatrix)
			# draw_confusion(confatrix)
			# tend = time.clock()
			# print(str(tend-time_init) + " is time taken")
		elif part =='b':
			gamma = 0.05
			penalty = 1
			tolerance = 1e-4
			print("tolerance,penalty,gamma for gaussian kernel for binary classification = " + str(tolerance) + "," + str(penalty) + "," + str(gamma))
			gaussian_kernel_soln = gaussian_kernel_cvxopt(train_data,train_output,gamma,penalty)
			(predicted,nSV) = gaussian_prediction_cvxopt(gaussian_kernel_soln,train_data,train_output,test_data,tolerance,gamma)
			print(str(nSV) + " support vectors")
			confatrix = confusion_matrix(test_output,predicted)
			print("Confusion Matrix")
			print(confatrix)
			# draw_confusion(confatrix)
			# tend = time.clock()
			# print(str(tend-time_init) + " is time taken")
		elif part == 'c':
			gamma = 0.05
			penalty = 1
			libsvm_both(train_data,train_output,test_data,test_output,gamma,penalty)
		else:
			print("No such part for binary classification")

	else:
		if part == 'a':
			gamma = 0.05
			penalty = 1
			tolerance = 1e-6
			print("tolerance value for gaussian kernel for multiclass classification= " + str(tolerance))
			t_start = time.clock()
			(test_output,prediction) = multiclass_svm_cvxopt(train_data_path,test_data_path,gamma,penalty,tolerance)
			t_end = time.clock()
			confatrix = confusion_matrix(test_output,prediction)
			print(confatrix)
			print("time taken by cvxopt multiclass= " + str(t_end-t_start))

		elif part =='b':
			gamma = 0.05
			penalty = 1
			(test_output,prediction) = multiclass_svm_libsvm(train_data_path,test_data_path,gamma,penalty)
			confatrix = confusion_matrix(test_output,prediction)
			print(confatrix)
			# draw_confusion(confatrix)
		elif part == 'd':
			gamma = 0.05
			penalty_array = [0.00001,0.001,1,5,10]
			validation_set_accuracy = np.zeros((1,5),dtype=float)
			test_accuracy = np.zeros((1,5),dtype=float)
			
			(train_data,train_output) = get_data(train_data_path,False,0,0)
			(test_data,test_output) = get_data(test_data_path,False,0,0)
			
			validation_data_X = train_data[18000:20000,:]
			validation_output_Y = train_output[18000:20000,:]
			training_data_X = train_data[0:18000,:]
			training_output_Y = train_output[0:18000,:]

			for i in range(len(penalty_array)):
				penalty = penalty_array[i]
				train_labels = []
				train_input = training_data_X.tolist()
				for i1 in range(training_output_Y.shape[0]):
					train_labels.append(training_output_Y[i1,0])

				validation_labels = []
				validation_input = validation_data_X.tolist()
				for i1 in range(validation_output_Y.shape[0]):
					validation_labels.append(validation_output_Y[i1,0])
				
				test_labels = []
				test_input = test_data.tolist()
				for j1 in range(test_output.shape[0]):
					test_labels.append(test_output[j1,0])

				problem = svm_problem(train_labels,train_input)
				gaussian_param = svm_parameter("-s 0 -c " + str(penalty) + " -t 2 -g " + str(gamma))
				gaussian_model = svm_train(problem,gaussian_param)
				svm_prediction_lbl,svm_prediction_acc,svm_prediction_val = svm_predict(test_labels,test_input,gaussian_model)
				test_accuracy[i] = svm_prediction_acc[0]
				svm_prediction_lbl,svm_prediction_acc,svm_prediction_val = svm_predict(validation_labels,validation_input,gaussian_model)
				validation_set_accuracy[i] = svm_prediction_acc[0]
			print("Validation Set Accuracy")
			print(validation_set_accuracy)
			print("Test set Accuracy")
			print(test_accuracy)
		else:
			print("No such part for multiclass classification")
	
	return

if __name__ == "__main__":
	main()