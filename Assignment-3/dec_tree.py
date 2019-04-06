import sys
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy.stats import entropy

class tree_Node:
	def __init__(self,datapoints_indices,parent,val=None,childs = [],num_nodes=1,feature_index=-1,answer=0):
		self.indices = datapoints_indices
		self.childs = childs
		self.parent = parent
		self.val = val
		self.feature_index = feature_index
		self.num_nodes = num_nodes
		self.answer = answer

def read_file(datapath,array_form):
	full_data = pd.read_csv(datapath)
	data_shape = full_data.shape
	full_data_arr = np.array(full_data.iloc[1:data_shape[0],:],dtype=int)
	
	if array_form==False:
		x = np.asmatrix(full_data_arr[:,1:data_shape[1]-1])
		y = np.asmatrix(full_data_arr[:,data_shape[1]-1:data_shape[1]])
		return (x,y)
	else:
		x = full_data_arr[:,1:data_shape[1]-1]
		y = full_data_arr[:,data_shape[1]-1:data_shape[1]]
		return (x,y)

def scikit_decision_tree(train_datapath,test_datapath,validation_datapath):
	(train_x,train_y) = read_file(train_datapath,True)
	(test_x,test_y) = read_file(test_datapath,True)
	(val_x,val_y) = read_file(validation_datapath,True)
	
	criteria = "my_entropy"
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

def pre_processing(mat,non_continuous_columns,neg_values_indices):
	median = np.asmatrix(np.median(mat,axis=0),dtype=int)
	new_data = np.multiply(np.asmatrix(np.ones(mat.shape,dtype=int)),mat>median)
	for i in non_continuous_columns:
		new_data[:,i] = mat[:,i]
	return new_data

def my_entropy(datapoints_indices,data_y):
	new_list = []
	for i in datapoints_indices:
		new_list.append(data_y[i,0])
	bin_count = np.divide(np.bincount(new_list),1.0*len(datapoints_indices))
	for j in range(len(bin_count)):
		if bin_count[j]==0:
			bin_count[j] = 1		
	return -1*np.sum(np.multiply(bin_count,np.log(bin_count)))

def split_parent(feature_index,datapoints_indices,data_x):
	a = np.unique(np.ravel(data_x[:,feature_index]))
	dicti = {}
	for i in datapoints_indices:
		for value in a:
			if data_x[i,feature_index]==value:
				if value in dicti:
					dicti[value].append(i)
				else:
					dicti[value] = [i]
				break
	return dicti

def info_gain(feature_index,datapoints_indices,data_x,data_y):
	dicti = split_parent(feature_index,datapoints_indices,data_x)
	ig = my_entropy(datapoints_indices,data_y) - (sum((len(dicti[prob]) * my_entropy(dicti[prob],data_y)) for prob in dicti.keys()))/len(datapoints_indices)
	return ig

def best_feature(datapoints_indices,data_x,data_y):
	max_gain = -1
	best_feature = -1
	for i in range(data_x.shape[1]):
		ig = info_gain(i,datapoints_indices,data_x,data_y)
		if ig>max_gain:
			max_gain = ig
			best_feature = i
	return (best_feature,max_gain)

def get_class(tree,data_point,train_y):
	if tree.feature_index==-1:
		return tree.answer
	else:
		for i in range(len(tree.childs)):
			if data_point[tree.feature_index]==tree.childs[i].val:
				j = i
				return get_class(tree.childs[j],data_point,train_y)
		if i==len(tree.childs)-1:
			new_list = []
			for i in tree.indices:
				new_list.append(train_y[i,0])
			return np.argmax(np.bincount(new_list))

def predict(tree,test_x,test_y):
	predicted = np.asmatrix(np.zeros((test_x.shape[0],1),dtype=int))
	for j in range(test_x.shape[0]):
		predicted[j,0] = get_class(tree,np.ravel(test_x[j,:]),test_y)
	return predicted

def grow_tree(train_x,train_y,datapoints_indices,parent):
	(bf,max_gain) = best_feature(datapoints_indices,train_x,train_y)
	node = tree_Node(datapoints_indices,parent,feature_index=bf)
	new_list = []
	for i in datapoints_indices:
		new_list.append(train_y[i,0])
	
	if len(set(new_list))<=1:
		node.feature_index = -1
		node.answer = np.argmax(np.bincount(new_list))
		return node
	
	if max_gain>=0:
		dicti = split_parent(bf,datapoints_indices,train_x)
		if len(dicti)==1:
			node.feature_index = -1
			new_list = []
			for i in dicti.values()[0]:
				new_list.append(train_y[i,0])
			node.answer = np.argmax(np.bincount(new_list))
			return node
		for val in dicti.keys():
			child = grow_tree(train_x,train_y,dicti[val],node)
			child.val = val
			node.childs.append(child)
			node.num_nodes+=child.num_nodes
		return node

def main():
	train_datapath = sys.argv[1]
	test_datapath = sys.argv[2]
	validation_datapath = sys.argv[3]
	
	(train_x,train_y) = read_file(train_datapath,False)
	(test_x,test_y) = read_file(test_datapath,False)
	(val_x,val_y) = read_file(validation_datapath,False)
	non_continuous_columns = [1,2,3,5,6,7,8,9,10]
	neg_values_indices = [5,6,7,8,9,10]
	train_x_new = pre_processing(train_x,non_continuous_columns,neg_values_indices)
	test_x_new = pre_processing(test_x,non_continuous_columns,neg_values_indices)
	validation_x_new = pre_processing(val_x,non_continuous_columns,neg_values_indices)

	datapoints_indices = []
	for i in range(train_x.shape[0]/100):
		datapoints_indices.append(i)
	tree = grow_tree(train_x_new,train_y,datapoints_indices,parent=None)
	print("Total Nodes = " + str(tree.num_nodes))
	
	predicted = predict(tree,test_x_new,train_y)
	print(confusion_matrix(test_y,predicted))
	print(accuracy_score(test_y,predicted))

	predicted = predict(tree,train_x_new,train_y)
	print(confusion_matrix(train_y,predicted))
	print(accuracy_score(train_y,predicted))

	predicted = predict(tree,validation_x_new,train_y)
	print(confusion_matrix(val_y,predicted))
	print(accuracy_score(val_y,predicted))

if __name__ == "__main__":
	main()