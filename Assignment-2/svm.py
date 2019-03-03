import csv
import pandas as pd
import numpy as np

def get_train_params():
	train_data = pd.read_csv('ass2_data_svm/train.csv',header=None,dtype=float)
	train_data = train_data.loc[(train_data[784] == 5) | (train_data[784] == 6)].values
	train_output = train_data[:,784:785]
	train_data = train_data[:,0:785]
	return (train_data,train_output)

def get_test_params():
	test_data = pd.read_csv('ass2_data_svm/test.csv',header=None,dtype=float).values
	test_data = test_data.loc[(test_data[784] == 5) | (test_data[784] == 6)].values
	test_output = test_data[:,784:785]
	test_data = test_data[:,0:785]
	return (test_data,test_output)

def main():
	(train_data,train_output) = get_train_params()
	(test_data,test_output) = get_test_params()
	
	return

if __name__ == "__main__":
	main()