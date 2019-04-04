import sys
import csv
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def generate_ohe(data_x,name):
	new_data = np.asmatrix(np.zeros((data_x.shape[0],85),dtype=int))
	for i in range(len(data_x)):
		for j in range(data_x.shape[1]/2):
			new_data[i,17*j + data_x[i,2*j]-1] = 1
			new_data[i,17*j + 4 + data_x[i,2*j+1]-1] = 1
	pd.DataFrame(np.array(new_data)).to_csv(name,header=None,index=None)

def scikit_nn(train_x,train_y,test_x,test_y,arch_details,num_iters,batchsize,activation,learning_rate):
	clf = MLPClassifier(alpha=0,shuffle=True,tol=0,warm_start=False,verbose=True,momentum=0,
		early_stopping=False,solver='sgd',hidden_layer_sizes=arch_details, random_state=1,
		activation=activation,max_iter=num_iters,batch_size=batchsize,learning_rate_init=learning_rate,learning_rate='constant')
	# clf = MLPClassifier(solver='sgd', alpha=1e-5,random_state=1,activation='relu')
	clf.fit(train_x,modify_output(train_y,10))
	predict = clf.predict(test_x)
	output = np.asmatrix(np.zeros((len(test_y),1)),dtype=int)
	for i in range(len(predict)):
		for j in range(10):
			if predict[i,j]==1:
				output[i,0] = j
	print(confusion_matrix(test_y,output))
	print(accuracy_score(test_y,output))
	# print(clf.coefs_)
	print(clf.get_params())

def read_file(datapath):
	full_data = pd.read_csv(datapath,header=None,dtype=int)
	data_shape = full_data.shape
	full_data_arr = np.array(full_data)
	x = np.asmatrix(full_data_arr[:,0:data_shape[1]-1])
	y = np.asmatrix(full_data_arr[:,data_shape[1]-1:data_shape[1]])
	return (x,y)
	
def sigmoid_activation(a):
	return 1/(1+np.exp(-1*a))
	
def sigmoid_derivative(a):
	return np.multiply(sigmoid_activation(a),1-sigmoid_activation(a))

def relu_activation(a):
	return np.multiply(a>0,a)

def relu_RHS_derivative(a):
	return np.multiply(a>=0,np.ones(a.shape,dtype=float))

def relu_LHS_derivative(a):
	return np.multiply(a>0,np.ones(a.shape,dtype=float))

def normalize_data_x(data_x):
	data_x = np.subtract(data_x,np.mean(data_x,axis=0))
	data_x = np.divide(data_x,np.var(data_x,axis=0))
	return data_x

def modify_output(data_y,num_classes):
	output_mat = np.asmatrix(np.zeros((len(data_y),num_classes),dtype=int))
	for i in range(len(data_y)):
		output_mat[i,data_y[i,0]]=1
	return output_mat
	
def initialize_params(arch_details):
	np.random.seed(1)
	params = {}
	for l in range(1,len(arch_details)):
		params["W"+str(l)] = np.random.randn(arch_details[l],arch_details[l-1])*0.01
		params["b"+str(l)] = np.zeros((arch_details[l],1),dtype=float)
	return params
	
def forward_prop(params,data_x,activation):
	forward_pass = {}
	num_layers = len(params)/2
	x = np.transpose(data_x)
	forward_pass["a0"] = x

	if activation=="logistic":
		for i in range(num_layers-1):
			x = np.dot(params["W"+str(i+1)],x) + params["b"+str(i+1)]
			forward_pass["z"+str(i+1)] = x
			x = sigmoid_activation(x)
			forward_pass["a"+str(i+1)] = x
	else:
		for i in range(num_layers-1):
			x = np.dot(params["W"+str(i+1)],x) + params["b"+str(i+1)]
			forward_pass["z"+str(i+1)] = x
			x = relu_activation(x)
			forward_pass["a"+str(i+1)] = x

	x = np.dot(params["W"+str(num_layers)],x) + params["b"+str(num_layers)]
	forward_pass["z"+str(num_layers)] = x
	x = sigmoid_activation(x)
	forward_pass["a"+str(num_layers)] = x
	return forward_pass

def loss_function(output_layer_output,actual_output):
	loss = np.multiply(output_layer_output-actual_output,output_layer_output-actual_output)
	return (-1*np.mean(loss,axis=1)/(output_layer_output.shape[1]))
	
def backward_prop(params,forward_pass,learning_rate,y_data,activation):
	der_dict = {}
	new_params = {}
	m = y_data.shape[1]
	der_output = np.multiply(forward_pass["a"+str(len(params)/2)] - y_data,sigmoid_derivative(forward_pass["z"+str(len(params)/2)]))
	# der_output = forward_pass["a"+str(len(params)/2)] - y_data
	der_dict["dZ"+str(len(params)/2)] = der_output
	
	if activation=="logistic":
		for i in xrange(len(params)/2 - 1,0,-1):
			der_output = np.multiply(np.dot(np.transpose(params["W"+str(i+1)]),der_dict["dZ"+str(i+1)]),sigmoid_derivative(forward_pass["z"+str(i)]))
			der_dict["dZ"+str(i)] = der_output

		for i in range(1,len(params)/2 +1):
			new_params["W"+str(i)] = params["W"+str(i)] - (learning_rate/m)*np.dot(der_dict["dZ"+str(i)],np.transpose(forward_pass["a"+str(i-1)]))
			new_params["b"+str(i)] = params["b"+str(i)] - (learning_rate/m)*np.sum(der_dict["dZ"+str(i)],axis=1)
	else:
		for i in xrange(len(params)/2 - 1,0,-1):
			der_output = np.multiply(np.dot(np.transpose(params["W"+str(i+1)]),der_dict["dZ"+str(i+1)]),relu_RHS_derivative(forward_pass["z"+str(i)]))
			der_dict["dZ"+str(i)] = der_output

		for i in range(1,len(params)/2 +1):
			new_params["W"+str(i)] = params["W"+str(i)] - (learning_rate/m)*np.dot(der_dict["dZ"+str(i)],np.transpose(forward_pass["a"+str(i-1)]))
			new_params["b"+str(i)] = params["b"+str(i)] - (learning_rate/m)*np.sum(der_dict["dZ"+str(i)],axis=1)
	return new_params

def prediction(params,data_x,activation):
	forward_pass = {}
	x = np.transpose(data_x)
	for i in range(len(params)/2):
		x = np.dot(params["W"+str(i+1)],x) + params["b"+str(i+1)]
		forward_pass["z"+str(i+1)] = x
		x = sigmoid_activation(x)
		forward_pass["a"+str(i+1)] = x
	output = np.exp(forward_pass["a"+str(len(params)/2)])
	summer = np.sum(output,axis=0)
	output = np.divide(output,summer)
	return np.argmax(output,axis=0)

def main():
	train_datapath = "dataset/neural_net/poker-hand-training-true.data"
	test_datapath = "dataset/neural_net/poker-hand-testing.data"

	# train_datapath = "../Assignment_2/mnist/train.csv"
	# test_datapath = "../Assignment_2/mnist/test.csv"
	
	(train_x,train_y) = read_file(train_datapath)
	(test_x,test_y) = read_file(test_datapath)
	# train_x = normalize_data_x(train_x)
	# test_x = normalize_data_x(test_x)
	
	# generate_ohe(train_x,"train_x.csv")
	# generate_ohe(test_x,"test_x.csv")

	train_x = np.asmatrix(pd.read_csv("train_x.csv",header=None,dtype=int))
	test_x = np.asmatrix(pd.read_csv("test_x.csv",header=None,dtype=int))

	learning_rate = 0.1
	architecture_details = [25]
	num_outputs = 10
	epochs = 2000
	batch_size = 100
	num_datapoints = len(train_x)
	modified_y = modify_output(train_y,num_outputs)
	error = 10
	epsilon = 1e-3
	
	neurons_list = [train_x.shape[1]]
	neurons_list.extend(architecture_details)
	neurons_list.append(num_outputs)
	activation = "logistic"
	# activation = "relu"
	params = initialize_params(neurons_list)
	# scikit_nn(train_x,train_y,test_x,test_y,architecture_details,epochs,batch_size,activation,learning_rate)
	cost_list = []
	counter = 0
	
	while error>epsilon:
		train_x = np.random.permutation(train_x)
		for batch_counter in range(num_datapoints/batch_size):
			begin = batch_counter*batch_size
			end = begin+batch_size
			if batch_counter==(num_datapoints/batch_size)-1:
				end = num_datapoints
			forward_pass  = forward_prop(params,train_x[begin:end,:],activation)
			loss_mag = loss_function(forward_pass["a"+str(len(params)/2)],np.transpose(modified_y[begin:end,:]))
			params = backward_prop(params,forward_pass,learning_rate,np.transpose(modified_y[begin:end,:]),activation)
		error = np.sqrt(np.dot(loss_mag,np.transpose(loss_mag))[0,0])
		print(str(counter) + " : " + str(error))
		cost_list.append(error)
		counter+=1
	
	predicted = np.transpose(prediction(params,test_x,activation))
	confatrix = confusion_matrix(test_y,predicted)
	print(confatrix)
	print(accuracy_score(test_y,predicted))
	plt.plot(cost_list)
	plt.show()

	
if __name__ == "__main__":
	main()