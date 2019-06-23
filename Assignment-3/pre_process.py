import sys
import csv
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def generate_ohe(data_x,name):
	new_data = np.asmatrix(np.zeros((data_x.shape[0],95),dtype=int))
	for i in range(len(data_x)):
		for j in range((int)(data_x.shape[1]-1)/2):
			new_data[i,17*j + data_x[i,2*j]-1] = 1
			new_data[i,17*j + 4 + data_x[i,2*j+1]-1] = 1
		new_data[i,85+data_x[i,-1]] = 1
	pd.DataFrame(np.array(new_data)).to_csv(name,header=None,index=None)

def readfile(datapath,name):
	data = pd.read_csv(datapath,header=None)
	datashape = data.shape
	data_array = np.asmatrix(np.array(data,dtype=int))
	generate_ohe(data_array,name)

def main():
	train_datapath = sys.argv[1]
	test_datapath = sys.argv[2]
	train_ohe_datapath = sys.argv[3]
	test_ohe_datapath = sys.argv[4]
	readfile(train_datapath,train_ohe_datapath)
	readfile(test_datapath,test_ohe_datapath)
	
if __name__ == "__main__":
	main()