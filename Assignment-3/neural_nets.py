import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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
	return np.multiply(sigmoid(a),1-sigmoid(a))

def relu_activation(a):
	return np.maximum(0,a)

def relu_derivative(a):
	return 0

def modify_output(data_y,num_classes):
	output_mat = np.asmatrix(np.zeros((len(data_y),num_classes),dtype=int))
	for i in range(len(data_y)):
		output_mat[i,data_y[i,0]]=1
	return output_mat
	
def initialize_params(arch_details):
	params = {}
	for l in range(1,len(arch_details)):
		params["W"+str(l)] = np.random.randn(arch_details[l],arch_details[l-1])*0.01
		params["b"+str(l)] = np.zeros((arch_details[l],1))
	return params
	
def forward_prop(params,data_x,activation):
	forward_pass = {}
	num_layers = len(params)/2
	x = np.transpose(data_x)
	for i in range(num_layers):
		x = np.dot(params["W"+str(i+1)],x) + params["b"+str(i+1)]
		forward_pass["z"+str(i+1)] = x
		x = sigmoid_activation(x)
		forward_pass["a"+str(i+1)] = x
	return forward_pass

def cost_function(output_layer_output,actual_output):
	loss = np.add(np.multiply(actual_output,np.log(output_layer_output)),np.multiply(1-actual_output,np.log(1-output_layer_output)))
	return (-1*np.sum(loss,axis=1)/(output_layer_output.shape[1]))
	
def backward_prop_sigmoid(params,cost,forward_pass,learning_rate):
	der_output = forward_pass["a"+str(len(params)/2)] - 
	return params

def backward_prop_relu(params,cost,forward_pass,learning_rate):
	der_output = forward_pass["a"+str(len(params)/2)] - 
	return params

def main():
	train_datapath = "dataset/neural_net/poker-hand-training-true.data"
	test_datapath = "dataset/neural_net/poker-hand-testing.data"
	(train_x,train_y) = read_file(train_datapath)
	(test_x,test_y) = read_file(test_datapath)
	
	learning_rate = 0.01
	architecture_details = [5]
	num_outputs = 10
	num_iters = 100
	modified_y = modify_output(train_y,num_outputs)

	neurons_list = [train_x.shape[1]]
	neurons_list.extend(architecture_details)
	neurons_list.append(num_outputs)
	
	params = initialize_params(neurons_list)
	for counter in range(num_iters):
		forward_pass  = forward_prop(params,train_x,activation)
		cost = cost_function(forward_pass["a"+str(len(params)/2)],np.transpose(modified_y))
		params = backward_prop_sigmoid(params,cost,forward_pass)
	print(params)
	
if __name__ == "__main__":
	main()