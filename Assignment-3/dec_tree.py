import sys
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def read_file(datapath,array_form):
	full_data = pd.read_csv(datapath)
	data_shape = full_data.shape
	full_data_arr = np.array(full_data.iloc[1:data_shape[0],:],dtype=int)
	
	if array_form==False:
		x = np.asmatrix(full_data_arr[:,0:data_shape[1]-1])
		y = np.asmatrix(full_data_arr[:,data_shape[1]-1:data_shape[1]])
		return (x,y)
	else:
		x = full_data_arr[:,0:data_shape[1]-1]
		y = full_data_arr[:,data_shape[1]-1:data_shape[1]]
		return (x,y)

def pre_processing(mat):
	# mean = np.mean(mat,axis=0,dtype=int)
	# non_continuous_columns = [0,2,3,4]
	# for i in non_continuous_columns:
	# 	mean[0,i] = 0
	# mat = np.subtract(mat,mean)
	return mat

def scikit_decision_tree(train_x,train_y,test_x,test_y,val_x,val_y):
	criteria = "entropy"
	split = "best"
	# best depth = 3
	depth = 3
	min_sample_leaf = 4
	min_sample_split = 2

	dec_tree = tree.DecisionTreeClassifier(criterion=criteria,splitter="best",max_depth=depth,min_samples_split=min_sample_split,min_samples_leaf=min_sample_leaf)
	dec_tree = dec_tree.fit(train_x,train_y)
	predicted = np.array(dec_tree.predict(val_x),dtype=int)
	confatrix = confusion_matrix(val_y,predicted)
	print(confatrix)
	print("Accuracy: " + str(accuracy_score(val_y,predicted)))

def one_hot_encoding(mat):
	return mat

def main():
	train_datapath = sys.argv[1]
	test_datapath = sys.argv[2]
	validation_datapath = sys.argv[3]

	(train_x,train_y) = read_file(train_datapath,True)
	(test_x,test_y) = read_file(test_datapath,True)
	(val_x,val_y) = read_file(validation_datapath,True)
	scikit_decision_tree(train_x,train_y,test_x,test_y,val_x,val_y)
	# x = pre_processing(x)


if __name__ == "__main__":
	main()