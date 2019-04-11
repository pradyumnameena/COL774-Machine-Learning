import sys
import collections
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

continuous_columns_global = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]

class tree_Node:
	def __init__(self,datapoints_indices,parent,median=0,val=-1,childs = [],num_nodes=1,feature_index=-1,answer=0):
		self.indices = datapoints_indices
		self.childs = childs
		self.parent = parent
		self.value = val
		self.feature_index = feature_index
		self.num_nodes = num_nodes
		self.answer = answer
		self.median = median

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

def ohe_func(mat,categorical_features,num_categories,negative_cols):
	for i in negative_cols:
		mat[:,i]+=2
	num_features = mat.shape[1]-len(categorical_features) + np.sum(num_categories)
	ret_mat = np.asmatrix(np.zeros((len(mat),num_features),dtype=int))
	return mat

def pre_processing(mat,non_continuous_columns,negative_cols):
	median = np.asmatrix(np.median(mat,axis=0),dtype=int)
	new_data = np.multiply(np.asmatrix(np.ones(mat.shape,dtype=int)),mat>=median)
	for i in non_continuous_columns:
		new_data[:,i] = mat[:,i]
	for i in negative_cols:
		new_data[:,i]+=2
	return new_data

def pre_processingV2(mat,negative_cols):
	for i in negative_cols:
		mat[:,i]+=2
	return mat

def my_entropy(datapoints_indices,data_y):
	new_list = np.ravel(data_y[datapoints_indices])
	bin_count = np.divide(np.bincount(new_list),1.0*len(datapoints_indices))
	bin_count = np.add(bin_count,np.multiply(np.ones(bin_count.shape,dtype=int),bin_count==0))
	return -1*np.sum(np.multiply(bin_count,np.divide(np.log(bin_count),np.log(2))))

def split_parent(feature_index,datapoints_indices,data_x):
	new_data2 = data_x[datapoints_indices,:][:,feature_index]
	dicti = {c: [datapoints_indices[x] for x in np.where(new_data2==c)[0]] for c in np.unique(np.ravel(new_data2))}
	return dicti
	
def info_gain(feature_index,datapoints_indices,data_x,data_y,modified):
	# change for part c
	dicti = {}
	if modified==True and feature_index in continuous_columns_global:
		curr_data = data_x[datapoints_indices,:][:,feature_index]
		median = np.median(curr_data,axis=0)[0,0]
		data_modified = np.multiply(np.asmatrix(np.ones(curr_data.shape,dtype=int)),curr_data>median)
		dicti = {c: [datapoints_indices[x] for x in np.where(data_modified==c)[0]] for c in np.unique(np.ravel(data_modified))}
	else:
		dicti = split_parent(feature_index,datapoints_indices,data_x)
	ig = 0.0
	for prob in dicti.keys():
		ig -= (np.divide(1.0*len(dicti[prob]),len(datapoints_indices)) * my_entropy(dicti[prob],data_y))
	return ig

def best_feature(datapoints_indices,data_x,data_y,modified):
	max_gain = -1
	best_feature = -1
	parent_entropy = my_entropy(datapoints_indices,data_y)
	for i in range(data_x.shape[1]):
		ig = parent_entropy + info_gain(i,datapoints_indices,data_x,data_y,modified)
		if ig>max_gain:
			max_gain = ig
			best_feature = i
	return best_feature,max_gain

def print_tree(tree):
	if tree.parent==None:
		print("Root Node. Feature Used -> " + str(tree.feature_index))
		new_list = []
		for c in tree.childs:
			new_list.append(c.value)
		print("Child Values -> " + str(new_list))
		for c in tree.childs:
			print_tree(c)
	else:
		print("Child splitted on feature -> " + str(tree.parent.feature_index))
		print("Value of child is -> " + str(tree.value))
		print("Feature to be used -> " + str(tree.feature_index))
		if tree.feature_index==-1:
			return
		new_list = []
		for c in tree.childs:
			new_list.append(c.value)
		print("Child Values -> " + str(new_list))
		for c in tree.childs:
			print_tree(c)

def get_class(tree,data_point,modified):
	if tree.feature_index==-1:
		return tree.answer
	else:
		if modified==False or (tree.feature_index not in continuous_columns_global):
			for i in range(len(tree.childs)):
				if data_point[tree.feature_index]==tree.childs[i].value:
					return get_class(tree.childs[i],data_point,modified)
		else:
			val = (data_point[tree.feature_index]>tree.median).astype(int)
			for i in range(len(tree.childs)):
				if tree.childs[i].value==val:
					return get_class(tree.childs[i],data_point,modified)
		return tree.answer

def predict(tree,test_x,modified):
	predicted = np.asmatrix(np.zeros((test_x.shape[0],1),dtype=int))
	for j in range(test_x.shape[0]):
		predicted[j,0] = get_class(tree,np.ravel(test_x[j,:]),modified)
	return predicted

def breadth_first_traversal(tree):
	node_list = []
	visited_list = [tree]
	while len(visited_list)!=0:
		vertex = visited_list.pop(0)
		node_list.append(vertex)
		for child in vertex.childs:
			visited_list.append(child)
	return node_list

def get_node_count(tree):
	if tree.feature_index==-1:
		return 1
	else:
		rv = 0
		for child in tree.childs:
			rv+= get_node_count(child)
		return rv+1

def grow_tree(train_x,train_y,datapoints_indices,parent=None):
	new_list = np.ravel(train_y[datapoints_indices])
	
	# PURE LEAF NODE || NO BEST FEATURE AVAILABLE
	if len(set(new_list)) <= 1:
		return tree_Node(datapoints_indices,parent,answer=np.argmax(np.bincount(new_list)),feature_index=-1)

	bf, max_gain = best_feature(datapoints_indices,train_x,train_y,False)
	if max_gain >= 0:
		dicti = split_parent(bf,datapoints_indices,train_x)
		node = tree_Node(datapoints_indices,parent,0,-1,[],1,feature_index=bf,answer=np.argmax(np.bincount(new_list)))

		if len(dicti)==1:
			return node
		for val in dicti.keys():
			child = grow_tree(train_x,train_y,dicti[val],parent=node)
			child.value = val
			node.childs.append(child)
			node.num_nodes+=child.num_nodes
		return node

def grow_treeV2(train_x,train_y,datapoints_indices,parent=None):
	new_list = np.ravel(train_y[datapoints_indices])
	
	# PURE LEAF NODE || NO BEST FEATURE AVAILABLE
	if len(set(new_list)) <= 1:
		return tree_Node(datapoints_indices,parent,answer=np.argmax(np.bincount(new_list)),feature_index=-1)

	bf, max_gain = best_feature(datapoints_indices,train_x,train_y,True)
	
	if max_gain >= 0:
		dicti = {}
		node = tree_Node(datapoints_indices,parent,0,-1,[],1,feature_index=bf,answer=np.argmax(np.bincount(new_list)))
		if bf in continuous_columns_global:
			new_data2 = (train_x[datapoints_indices,:][:,bf])
			median = np.median(new_data2,axis=0)[0,0]
			new_data2 = np.multiply(np.asmatrix(np.ones(new_data2.shape,dtype=int)),new_data2>median)
			dicti = {c: [datapoints_indices[x] for x in np.where(new_data2==c)[0]] for c in np.unique(np.ravel(new_data2))}
			node.median = median
		else:
			dicti = split_parent(bf,datapoints_indices,train_x)
		if len(dicti)==1:
			return node
		for val in dicti.keys():
			child = grow_treeV2(train_x,train_y,dicti[val],parent=node)
			child.value = val
			node.childs.append(child)
			node.num_nodes+=child.num_nodes
		return node

def main():
	part_num = (int)(sys.argv[1])
	train_datapath = sys.argv[2]
	test_datapath = sys.argv[3]
	validation_datapath = sys.argv[4]

	if part_num==1:
		(train_x,train_y) = read_file(train_datapath,False)
		(test_x,test_y) = read_file(test_datapath,False)
		(val_x,val_y) = read_file(validation_datapath,False)
		
		non_continuous_columns = [1,2,3,5,6,7,8,9,10]
		negative_cols = [5,6,7,8,9,10]
		train_x_new = pre_processing(train_x,non_continuous_columns,negative_cols)
		test_x_new = pre_processing(test_x,non_continuous_columns,negative_cols)
		validation_x_new = pre_processing(val_x,non_continuous_columns,negative_cols)

		datapoints_indices = []
		for i in range(train_x.shape[0]):
			datapoints_indices.append(i)
		
		my_tree = grow_tree(train_x_new,train_y,datapoints_indices)
		# print_tree(tree)
		print("Total Nodes -> " + str(get_node_count(my_tree)))
		modified = False
		print("Training Data")
		predicted = predict(my_tree,train_x_new,modified)
		print(confusion_matrix(train_y,predicted))
		print(accuracy_score(train_y,predicted))

		print("Testing Data")
		predicted = predict(my_tree,test_x_new,modified)
		print(confusion_matrix(test_y,predicted))
		print(accuracy_score(test_y,predicted))

		print("Validation Data")
		predicted = predict(my_tree,validation_x_new,modified)
		print(confusion_matrix(val_y,predicted))
		print(accuracy_score(val_y,predicted))
	
	elif part_num==2:
		(train_x,train_y) = read_file(train_datapath,False)
		(test_x,test_y) = read_file(test_datapath,False)
		(val_x,val_y) = read_file(validation_datapath,False)
		
		non_continuous_columns = [1,2,3,5,6,7,8,9,10]
		negative_cols = [5,6,7,8,9,10]
		train_x_new = pre_processing(train_x,non_continuous_columns,negative_cols)
		test_x_new = pre_processing(test_x,non_continuous_columns,negative_cols)
		validation_x_new = pre_processing(val_x,non_continuous_columns,negative_cols)

		datapoints_indices = []
		for i in range(train_x.shape[0]):
			datapoints_indices.append(i)
		
		my_tree = grow_tree(train_x_new,train_y,datapoints_indices)
		
		node_list = breadth_first_traversal(my_tree)
		train_accuracy = {}
		validation_accuracy = {}
		test_accuracy = {}
		node_count_dict = {}

		modified = False
		train_accuracy[0] = accuracy_score(train_y,predict(my_tree,train_x_new,modified))
		validation_accuracy[0] = accuracy_score(val_y,predict(my_tree,validation_x_new,modified))
		test_accuracy[0] = accuracy_score(test_y,predict(my_tree,test_x_new,modified))
		node_count_dict[0] = get_node_count(my_tree)

		iter_num = 0
		best_accuracy = -1

		while True:
			acc_prev = accuracy_score(val_y,predict(my_tree,validation_x_new,modified))
			acc_after = 0
			best_node = tree
			iter_num+=1
			node_list.reverse()
			counter = 0
			# print("Iteration -> " + str(iter_num))
			for node in node_list:
				# print(counter)
				if node.feature_index!=-1:
					node_childs = node.childs
					node_feature = node.feature_index
					node.childs = []
					node.feature_index = -1
					acc_after = accuracy_score(val_y,predict(my_tree,validation_x_new,modified))
					if acc_after>best_accuracy:
						# print(acc_after)
						best_accuracy = acc_after
						best_node = node
					node.feature_index = node_feature
					node.childs = node_childs
				counter+=1
			if best_accuracy>acc_prev:
				best_node.childs = []
				best_node.feature_index = -1
				node_count = get_node_count(my_tree)
				node_count_dict[iter_num] = node_count
				train_accuracy[iter_num] = accuracy_score(val_y,predict(my_tree,validation_x_new,modified))
				test_accuracy[iter_num] = accuracy_score(val_y,predict(my_tree,validation_x_new,modified))
				validation_accuracy[iter_num] = accuracy_score(val_y,predict(my_tree,validation_x_new,modified))
				node_list = breadth_first_traversal(my_tree)
			else:
				break

		# print_tree(tree)
		print("Total Nodes -> " + str(get_node_count(my_tree)))
		print("Training Data")
		predicted = predict(my_tree,train_x_new,modified)
		print(confusion_matrix(train_y,predicted))
		print(accuracy_score(train_y,predicted))

		print("Testing Data")
		predicted = predict(my_tree,test_x_new,modified)
		print(confusion_matrix(test_y,predicted))
		print(accuracy_score(test_y,predicted))

		print("Validation Data")
		predicted = predict(my_tree,validation_x_new,modified)
		print(confusion_matrix(val_y,predicted))
		print(accuracy_score(val_y,predicted))

		fig = plt.figure()
		plt.title("Accuracies vs Number of Nodes")
		plt.plot(node_count_dict.values(), test_accuracy.values(), label = 'Testing')
		plt.plot(node_count_dict.values(), validation_accuracy.values(), label = 'Validation')
		plt.plot(node_count_dict.values(), train_accuracy.values(), label = 'Training')
		plt.xlabel("Number of Nodes")
		plt.ylabel('Accuracies')
		plt.legend()
		# plt.show()
		fig.savefig("Accuracy_with_Nodes"+'.png')

	elif part_num==3:
		(train_x,train_y) = read_file(train_datapath,False)
		(test_x,test_y) = read_file(test_datapath,False)
		(val_x,val_y) = read_file(validation_datapath,False)
		
		negative_cols = [5,6,7,8,9,10]
		train_x_new = pre_processingV2(train_x,negative_cols)
		test_x_new = pre_processingV2(test_x,negative_cols)
		validation_x_new = pre_processingV2(val_x,negative_cols)

		datapoints_indices = []
		for i in range(train_x.shape[0]):
			datapoints_indices.append(i)
		
		my_tree = grow_treeV2(train_x_new,train_y,datapoints_indices)
		print("Total Nodes = " + str(get_node_count(my_tree)))
		# print_tree(my_tree)

		modified = True
		print("Training Data")
		predicted = predict(my_tree,train_x_new,modified)
		print(confusion_matrix(train_y,predicted))
		print(accuracy_score(train_y,predicted))

		print("Testing Data")
		predicted = predict(my_tree,test_x_new,modified)
		print(confusion_matrix(test_y,predicted))
		print(accuracy_score(test_y,predicted))

		print("Validation Data")
		predicted = predict(my_tree,validation_x_new,modified)
		print(confusion_matrix(val_y,predicted))
		print(accuracy_score(val_y,predicted))

	elif part_num==4:
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
		
		print("Training Set")
		predicted_train = np.array(dec_tree.predict(train_x),dtype=int)
		confatrix_train = confusion_matrix(train_y,predicted_train)
		print(confatrix_train)
		print("Accuracy: " + str(accuracy_score(train_y,predicted_train)))

		print("Test Set")
		predicted_test = np.array(dec_tree.predict(test_x),dtype=int)
		confatrix_test = confusion_matrix(test_y,predicted_test)
		print(confatrix_test)
		print("Accuracy: " + str(accuracy_score(test_y,predicted_test)))

		print("Validation Set")
		predicted_val = np.array(dec_tree.predict(val_x),dtype=int)
		confatrix_val = confusion_matrix(val_y,predicted_val)
		print(confatrix_val)
		print("Accuracy: " + str(accuracy_score(val_y,predicted_val)))

	elif part_num==5:
		categorical_features = [1,2,3,5,6,7,8,9,10]
		num_categories = [4,4,4,4,4,4,4,4,4]
		negative_cols = [5,6,7,8,9,10]
		(train_x,train_y) = read_file(train_datapath,True)
		(test_x,test_y) = read_file(test_datapath,True)
		(val_x,val_y) = read_file(validation_datapath,True)

		train_x_new = ohe_func(train_x,categorical_features,num_categories,negative_cols)
		test_x_new = ohe_func(test_x,categorical_features,num_categories,negative_cols)
		val_x_new = ohe_func(val_x,categorical_features,num_categories,negative_cols)

		criteria = "gini"
		depth = 10
		min_sample_leaf = 1
		min_sample_split = 2
		# dec_tree = tree.DecisionTreeClassifier(criterion=criteria,splitter="best",max_depth=depth,min_samples_split=min_sample_split,min_samples_leaf=min_sample_leaf)
		dec_tree = tree.DecisionTreeClassifier(criterion=criteria)
		dec_tree = dec_tree.fit(train_x_new,train_y)

		print("Training Set")
		predicted_train = np.array(dec_tree.predict(train_x_new),dtype=int)
		confatrix_train = confusion_matrix(train_y,predicted_train)
		print(confatrix_train)
		print("Accuracy: " + str(accuracy_score(train_y,predicted_train)))

		print("Test Set")
		predicted_test = np.array(dec_tree.predict(test_x_new),dtype=int)
		confatrix_test = confusion_matrix(test_y,predicted_test)
		print(confatrix_test)
		print("Accuracy: " + str(accuracy_score(test_y,predicted_test)))

		print("Validation Set")
		predicted_val = np.array(dec_tree.predict(val_x_new),dtype=int)
		confatrix_val = confusion_matrix(val_y,predicted_val)
		print(confatrix_val)
		print("Accuracy: " + str(accuracy_score(val_y,predicted_val)))

	elif part_num==6:
		categorical_features = [1,2,3,5,6,7,8,9,10]
		num_categories = [4,4,4,4,4,4,4,4,4]
		negative_cols = [5,6,7,8,9,10]
		(train_x,train_y) = read_file(train_datapath,True)
		(test_x,test_y) = read_file(test_datapath,True)
		(val_x,val_y) = read_file(validation_datapath,True)

		train_x_new = ohe_func(train_x,categorical_features,num_categories,negative_cols)
		test_x_new = ohe_func(test_x,categorical_features,num_categories,negative_cols)
		val_x_new = ohe_func(val_x,categorical_features,num_categories,negative_cols)

		num_estimators = 10
		criteria = "entropy"
		depth = 10
		min_sample_leaf = 1
		min_sample_split = 2
		bs = True
		# rf = RandomForestClassifier(n_estimators=num_estimators,criterion=criteria,max_depth=depth,min_samples_split=min_sample_split,min_samples_leaf=min_sample_leaf,bootstrap=bs)
		rf = RandomForestClassifier(criterion=criteria)
		rf.fit(train_x_new,np.ravel(train_y))

		print("Training Set")
		predicted_train = np.array(rf.predict(train_x_new),dtype=int)
		confatrix_train = confusion_matrix(train_y,predicted_train)
		print(confatrix_train)
		print("Accuracy: " + str(accuracy_score(train_y,predicted_train)))

		print("Test Set")
		predicted_test = np.array(rf.predict(test_x_new),dtype=int)
		confatrix_test = confusion_matrix(test_y,predicted_test)
		print(confatrix_test)
		print("Accuracy: " + str(accuracy_score(test_y,predicted_test)))

		print("Validation Set")
		predicted_val = np.array(rf.predict(val_x_new),dtype=int)
		confatrix_val = confusion_matrix(val_y,predicted_val)
		print(confatrix_val)
		print("Accuracy: " + str(accuracy_score(val_y,predicted_val)))

	else:
		print("Invalid part")

if __name__ == "__main__":
	main()