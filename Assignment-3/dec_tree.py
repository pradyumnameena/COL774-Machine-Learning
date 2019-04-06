import sys
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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

def pre_processing(mat,non_continuous_columns,neg_values_indices,amount):
	median = np.asmatrix(np.median(mat,axis=0),dtype=int)
	new_data = np.multiply(np.asmatrix(np.ones(mat.shape,dtype=int)),mat>median)
	for i in non_continuous_columns:
		new_data[:,i] = mat[:,i]
	for i in neg_values_indices:
		new_data[:,i]+=amount
	return new_data

def entropy(datapoints_indices,data_y):
	new_list = []
	for i in datapoints_indices:
		new_list.append(data_y[i,0])
	bin_count = np.divide(np.bincount(new_list),1.0*len(datapoints_indices))
	for j in range(len(bin_count)):
		if bin_count[j]==0:
			bin_count[j] = 1
	return -1*np.sum(np.multiply(bin_count,np.divide(np.log(bin_count),np.log(2))))

def split_parent(feature_index,datapoints_indices,data_x):
	a = np.unique(np.ravel(data_x[:,feature_index]))
	dicti = {}
	for value in a:
		for i in datapoints_indices:
			if data_x[i,feature_index]==value:
				if value in dicti:
					dicti[value].append(i)
				else:
					dicti[value] = [i]
	return dicti

def best_feature(datapoints_indices,data_x,data_y):
	max_gain = -1.5
	best_feature = -1
	for i in range(data_x.shape[1]):
		dicti = split_parent(i,datapoints_indices,data_x)
		info_gain = entropy(datapoints_indices,data_y) - sum(((len(dicti[prob]) * entropy(dicti[prob],data_y))/len(datapoints_indices)) for prob in dicti.keys())
		if info_gain>max_gain:
			max_gain = info_gain
			best_feature = i
	# print(best_feature)
	# print(max_gain)
	return (best_feature,max_gain)

def grow_tree(data_x,data_y,datapoints_indices,parent = None):
	(bf,info_gain) = best_feature(datapoints_indices,data_x,data_y)
	if bf !=-1 and info_gain>0:
		parent_node = tree_Node(datapoints_indices,parent,feature_index=bf)
		dicti = split_parent(bf,datapoints_indices,data_x)
		# print(bf,info_gain)
		if len(dicti)==1:
			parent_node.answer = dicti.keys()[0]
			return parent_node
		for val in dicti:
			child = grow_tree(data_x,data_y,dicti[val],parent_node)
			child.val = val
			parent_node.childs.append(child)
			parent_node.num_nodes+=child.num_nodes
		return parent_node
	else:
		new_list = []
		for i in datapoints_indices:
			new_list.append(data_y[i,0])
		ans = np.argmax(np.bincount(new_list))
		return tree_Node(datapoints_indices,None,answer=ans)

def get_class(tree,data_point):
	if tree.feature_index!=-1:
		for i in range(len(tree.childs)):
			if tree.childs[i].val == data_point[0,tree.childs[i].feature_index]:
				return get_class(tree.childs[i],data_point)
	else:
		return tree.answer

def predict(tree,test_x):
	predicted = np.asmatrix(np.zeros((len(test_x),1),dtype=int))
	for i in range(len(test_x)):
		predicted[i,0] = get_class(tree,test_x[i,:])
	return predicted

def main():
	train_datapath = sys.argv[1]
	test_datapath = sys.argv[2]
	validation_datapath = sys.argv[3]
	
	(train_x,train_y) = read_file(train_datapath,False)
	(test_x,test_y) = read_file(test_datapath,False)
	(val_x,val_y) = read_file(validation_datapath,False)
	non_continuous_columns = [1,2,3,5,6,7,8,9,10]
	neg_values_indices = [5,6,7,8,9,10]
	amount = 2
	train_x_new = pre_processing(train_x,non_continuous_columns,neg_values_indices,amount)
	test_x_new = pre_processing(test_x,non_continuous_columns,neg_values_indices,amount)
	
	datapoints_indices = []
	for i in range(train_x.shape[0]):
		datapoints_indices.append(i)
	tree = grow_tree(train_x_new,train_y,datapoints_indices)
	print("Total Nodes = " + str(tree.num_nodes))
	predicted = predict(tree,test_x_new)
	confatrix = confusion_matrix(test_y,predicted)
	print(confatrix)
	print(accuracy_score(test_y,predicted))

if __name__ == "__main__":
	main()