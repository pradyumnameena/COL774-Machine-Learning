import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

def scikit_nn(train_x,train_y,test_x,test_y,arch_details,num_iters):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=arch_details, random_state=1,activation='relu',max_iter=num_iters)
	clf.fit(train_x,modify_output(train_y,10))
	print(clf.get_params(deep=True))
	predict = clf.predict(test_x)
	output = np.asmatrix(np.zeros((len(test_y),1)),dtype=int)
	for i in range(len(predict)):
		for j in range(10):
			if predict[i,j]==1:
				output[i,0] = j
	print(confusion_matrix(test_y,output))
	print(accuracy_score(test_y,predicted))

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
	return np.multiply(a,a>0)

def relu_RHS_derivative(a):
	return np.multiply(a>=0,np.ones(a.shape,dtype=float))

def relu_LHS_derivative(a):
	return np.multiply(a>0,np.ones(a.shape,dtype=float))

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
	
	if activation=="sigmoid":
		for i in range(num_layers):
			x = np.dot(params["W"+str(i+1)],x) + params["b"+str(i+1)]
			forward_pass["z"+str(i+1)] = x
			x = sigmoid_activation(x)
			forward_pass["a"+str(i+1)] = x
	else:
		for i in range(num_layers):
			x = np.dot(params["W"+str(i+1)],x) + params["b"+str(i+1)]
			forward_pass["z"+str(i+1)] = x
			x = relu_activation(x)
			forward_pass["a"+str(i+1)] = x
	return forward_pass

def cost_function(output_layer_output,actual_output):
	loss = np.add(np.multiply(actual_output,np.log(output_layer_output)),np.multiply(1-actual_output,np.log(1-output_layer_output)))
	return (-1*np.sum(loss,axis=1)/(output_layer_output.shape[1]))
	
def backward_prop(params,cost,forward_pass,learning_rate,y_data,activation):
	der_dict = {}
	new_params = {}
	m = y_data.shape[1]
	
	if activation=="sigmoid":
		der_output = forward_pass["a"+str(len(params)/2)] - y_data
		der_dict["dZ"+str(len(params)/2)] = der_output
		for i in xrange(len(params)/2 - 1,0,-1):
			der_output = np.multiply(np.dot(np.transpose(params["W"+str(i+1)]),der_dict["dZ"+str(i+1)]),sigmoid_derivative(forward_pass["z"+str(i)]))
			der_dict["dZ"+str(i)] = der_output

		for i in range(1,len(params)/2 +1):
			new_params["W"+str(i)] = params["W"+str(i)] - (learning_rate/m)*np.dot(der_dict["dZ"+str(i)],np.transpose(forward_pass["a"+str(i-1)]))
			new_params["b"+str(i)] = params["b"+str(i)] - (learning_rate/m)*np.sum(der_dict["dZ"+str(i)],axis=1)
	else:
		der_output = forward_pass["a"+str(len(params)/2)] - y_data
		der_dict["dZ"+str(len(params)/2)] = der_output
		for i in xrange(len(params)/2 - 1,0,-1):
			der_output = np.multiply(np.dot(np.transpose(params["W"+str(i+1)]),der_dict["dZ"+str(i+1)]),sigmoid_derivative(forward_pass["z"+str(i)]))
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
	(train_x,train_y) = read_file(train_datapath)
	(test_x,test_y) = read_file(test_datapath)
	
	learning_rate = 0.001
	architecture_details = [10,10]
	num_outputs = 10
	num_iters = 100
	modified_y = modify_output(train_y,num_outputs)

	neurons_list = [train_x.shape[1]]
	neurons_list.extend(architecture_details)
	neurons_list.append(num_outputs)
	# scikit_nn(train_x,train_y,test_x,test_y,architecture_details,num_iters)
	# activation = "sigmoid"
	activation = "relu"
	params = initialize_params(neurons_list)

	for counter in range(num_iters):
		if counter%100==0:
			print(counter)
		forward_pass  = forward_prop(params,train_x,activation)
		cost = cost_function(forward_pass["a"+str(len(params)/2)],np.transpose(modified_y))
		params = backward_prop(params,cost,forward_pass,learning_rate,np.transpose(modified_y),activation)
	
	predicted = np.transpose(prediction(params,test_x,activation))
	confatrix = confusion_matrix(test_y,predicted)
	print(confatrix)
	print(accuracy_score(test_y,predicted))
	
if __name__ == "__main__":
	main()