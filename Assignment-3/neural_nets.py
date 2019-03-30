import sys
import numpy as np
import pandas as pd
from sklearn import tree
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

def relu_activation(a):
	return max(0,a)

def main():
	train_datapath = "dataset/neural_net/poker-hand-training-true.data"
	test_datapath = "dataset/neural_net/poker-hand-testing.data"
	(train_x,train_y) = read_file(train_datapath)
	(test_x,test_y) = read_file(test_datapath)

	learning_rate = 0.01
	architecture_details = [5,4]

if __name__ == "__main__":
	main()