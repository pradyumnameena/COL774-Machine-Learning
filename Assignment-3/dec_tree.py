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

part1_train_accuracy = {}
part1_test_accuracy = {}
part1_val_accuracy = {}

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

# Reading the file
def read_file(datapath):
	full_data = pd.read_csv(datapath)
	data_shape = full_data.shape
	full_data_arr = np.array(full_data.iloc[1:data_shape[0],:],dtype=int)
	x = full_data_arr[:,1:data_shape[1]-1]
	y = full_data_arr[:,data_shape[1]-1:data_shape[1]]
	return (x,y)

# generates one hot encoding
def ohe_func(mat,categorical_features,num_categories,negative_cols):
	for i in negative_cols:
		mat[:,i]+=2
	num_features = mat.shape[1]-len(categorical_features) + np.sum(num_categories)
	ret_mat = np.asmatrix(np.zeros((len(mat),num_features),dtype=int))
	new_index = 0
	cat_index = 0
	for i in range(mat.shape[1]):
		if i in categorical_features:
			for j in range(mat.shape[0]):
				ret_mat[j,new_index+mat[j,i]] = 1
			new_index+=num_categories[cat_index]
			cat_index+=1
		else:
			ret_mat[:,new_index] = np.reshape(mat[:,i],(len(ret_mat),1))
			new_index+=1
	return ret_mat

# preprcessing based on median (1 if x>=median else 0) for continuous columns
def pre_processing(mat,non_continuous_columns,negative_cols):
	median = np.asmatrix(np.median(mat,axis=0),dtype=int)
	new_data = np.multiply(np.asmatrix(np.ones(mat.shape,dtype=int)),mat>=median)
	for i in non_continuous_columns:
		new_data[:,i] = mat[:,i]
	for i in negative_cols:
		new_data[:,i]+=2
	return new_data

# To handle negative columns
def pre_processingV2(mat,negative_cols):
	for i in negative_cols:
		mat[:,i]+=2
	return mat

# Entropy function
def my_entropy(datapoints_indices,data_y):
	new_list = np.ravel(data_y[datapoints_indices])
	bin_count = np.divide(np.bincount(new_list),1.0*len(datapoints_indices))
	bin_count = np.add(bin_count,np.multiply(np.ones(bin_count.shape,dtype=int),bin_count==0))
	return -1*np.sum(np.multiply(bin_count,np.divide(np.log(bin_count),np.log(2))))

# Split function based on number of unique values of feature_index
def split_parent(feature_index,datapoints_indices,data_x):
	new_data2 = data_x[datapoints_indices,:][:,feature_index]
	dicti = {c: [datapoints_indices[x] for x in np.where(new_data2==c)[0]] for c in np.unique(np.ravel(new_data2))}
	return dicti

# Information gain function
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

# Selecting the best feature and its information gain
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

# Function to print tree
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

# Prediction function for a single point
def get_class(tree,data_point,current_depth,allowed_depth,modified,max_depth):
	if tree.feature_index==-1 or current_depth==allowed_depth or current_depth==max_depth:
		return tree.answer
	else:
		if modified==False or (tree.feature_index not in continuous_columns_global):
			for i in range(len(tree.childs)):
				if data_point[tree.feature_index]==tree.childs[i].value:
					return get_class(tree.childs[i],data_point,current_depth+1,allowed_depth,modified,max_depth)
		else:
			val = (data_point[tree.feature_index]>tree.median).astype(int)
			for i in range(len(tree.childs)):
				if tree.childs[i].value==val:
					return get_class(tree.childs[i],data_point,current_depth+1,allowed_depth,modified,max_depth)
		return tree.answer

# Overall prediction
def predict(tree,test_x,modified,allowed_depth,max_depth):
	predicted = np.asmatrix(np.zeros((test_x.shape[0],1),dtype=int))
	for j in range(test_x.shape[0]):
		predicted[j,0] = get_class(tree,np.ravel(test_x[j,:]),0,allowed_depth,modified,max_depth)
	return predicted

# BFS traversal for getting all nodes of the tree
def breadth_first_traversal(tree):
	node_list = []
	visited_list = [tree]
	while len(visited_list)!=0:
		vertex = visited_list.pop(0)
		node_list.append(vertex)
		for child in vertex.childs:
			visited_list.append(child)
	return node_list

# getting height of the tree
def get_max_depth(tree):
	if tree.feature_index==-1 or len(tree.childs)==0:
		return 1
	else:
		child_depths = [get_max_depth(child) for child in tree.childs]
		return 1+max(child_depths)

# getting number of nodes in the tree
def get_node_count(tree):
	if tree.feature_index==-1:
		return 1
	else:
		child_node_count = [get_node_count(child) for child in tree.childs]
		return 1+sum(child_node_count)

# Functions to grow the tree depending on the part
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
	# Taking input from console
	part_num = (int)(sys.argv[1])
	train_datapath = sys.argv[2]
	test_datapath = sys.argv[3]
	validation_datapath = sys.argv[4]

	# Reading the dataset
	(train_x,train_y) = read_file(train_datapath,False)
	(test_x,test_y) = read_file(test_datapath,False)
	(val_x,val_y) = read_file(validation_datapath,False)
	if(part_num>=4):
		train_x = np.asmatrix(train_x)
		train_y = np.asmatrix(train_y)
		test_x = np.asmatrix(test_x)
		test_y = np.asmatrix(test_y)
		val_x = np.asmatrix(val_x)
		val_y = np.asmatrix(val_y)
	
	if part_num==1:
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
		max_depth = get_max_depth(my_tree)+1
		
		print("Total Nodes -> " + str(get_node_count(my_tree)))
		modified = False
		print("Training Data")
		predicted = predict(my_tree,train_x_new,modified,max_depth,max_depth)
		print(confusion_matrix(train_y,predicted))
		print(accuracy_score(train_y,predicted))

		print("Testing Data")
		predicted = predict(my_tree,test_x_new,modified,max_depth,max_depth)
		print(confusion_matrix(test_y,predicted))
		print(accuracy_score(test_y,predicted))

		print("Validation Data")
		predicted = predict(my_tree,validation_x_new,modified,max_depth,max_depth)
		print(confusion_matrix(val_y,predicted))
		print(accuracy_score(val_y,predicted))

		# Getting accuracies
		for depth in range(max_depth):
			part1_train_accuracy[depth] = accuracy_score(train_y,predict(my_tree,train_x_new,modified,depth,max_depth))
			part1_test_accuracy[depth] = accuracy_score(test_y,predict(my_tree,test_x_new,modified,depth,max_depth))
			part1_val_accuracy[depth] = accuracy_score(val_y,predict(my_tree,validation_x_new,modified,depth,max_depth))

		# print(part1_val_accuracy)
		# print(part1_train_accuracy)
		# print(part1_test_accuracy)

		fig = plt.figure()
		plt.title("Accuracies vs Depth")
		plt.plot(part1_test_accuracy.keys(),part1_test_accuracy.values(),label = 'Testing')
		plt.plot(part1_val_accuracy.keys(),part1_val_accuracy.values(), label = 'Validation')
		plt.plot(part1_train_accuracy.keys(),part1_train_accuracy.values(), label = 'Training')
		plt.xlabel("Depth")
		plt.ylabel("Accuracies")
		plt.legend()
		# plt.show()
		fig.savefig("Graph_Part1"+'.png')	
	
	elif part_num==2:
		non_continuous_columns = [1,2,3,5,6,7,8,9,10]
		negative_cols = [5,6,7,8,9,10]
		train_x_new = pre_processing(train_x,non_continuous_columns,negative_cols)
		test_x_new = pre_processing(test_x,non_continuous_columns,negative_cols)
		validation_x_new = pre_processing(val_x,non_continuous_columns,negative_cols)

		datapoints_indices = []
		for i in range(train_x.shape[0]):
			datapoints_indices.append(i)
		
		my_tree = grow_tree(train_x_new,train_y,datapoints_indices)
		max_depth = get_max_depth(my_tree)+1
		node_list = breadth_first_traversal(my_tree)
		train_accuracy = {}
		validation_accuracy = {}
		test_accuracy = {}
		node_count_dict = {}

		modified = False
		train_accuracy[0] = accuracy_score(train_y,predict(my_tree,train_x_new,modified,max_depth,max_depth))
		validation_accuracy[0] = accuracy_score(val_y,predict(my_tree,validation_x_new,modified,max_depth,max_depth))
		test_accuracy[0] = accuracy_score(test_y,predict(my_tree,test_x_new,modified,max_depth,max_depth))
		node_count_dict[0] = get_node_count(my_tree)
		# print(node_count_dict[0])
		iter_num = 0
		best_accuracy = -1
		while True:
			acc_prev = accuracy_score(val_y,predict(my_tree,validation_x_new,modified,max_depth,max_depth))
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
					acc_after = accuracy_score(val_y,predict(my_tree,validation_x_new,modified,max_depth,max_depth))
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
				train_accuracy[iter_num] = accuracy_score(train_y,predict(my_tree,train_x_new,modified,max_depth,max_depth))
				test_accuracy[iter_num] = accuracy_score(test_y,predict(my_tree,test_x_new,modified,max_depth,max_depth))
				validation_accuracy[iter_num] = accuracy_score(val_y,predict(my_tree,validation_x_new,modified,max_depth,max_depth))
				node_list = breadth_first_traversal(my_tree)
				print(get_node_count(my_tree))
			else:
				break

		# print_tree(tree)
		print("Total Nodes -> " + str(get_node_count(my_tree)))
		print("Training Data")
		predicted = predict(my_tree,train_x_new,modified,max_depth,max_depth)
		print(confusion_matrix(train_y,predicted))
		print(accuracy_score(train_y,predicted))

		print("Testing Data")
		predicted = predict(my_tree,test_x_new,modified,max_depth,max_depth)
		print(confusion_matrix(test_y,predicted))
		print(accuracy_score(test_y,predicted))

		print("Validation Data")
		predicted = predict(my_tree,validation_x_new,modified,max_depth,max_depth)
		print(confusion_matrix(val_y,predicted))
		print(accuracy_score(val_y,predicted))

		fig = plt.figure()
		plt.title("Accuracies vs Number of Nodes")
		plt.plot(node_count_dict.keys(), test_accuracy.values(), label = 'Testing')
		plt.plot(node_count_dict.keys(), validation_accuracy.values(), label = 'Validation')
		plt.plot(node_count_dict.keys(), train_accuracy.values(), label = 'Training')
		plt.xlabel("Number of Nodes")
		plt.ylabel('Accuracies')
		plt.legend()
		# plt.show()
		fig.savefig("Graph_Part2"+'.png')

	elif part_num==3:
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
		max_depth = get_max_depth(my_tree)+1
		modified = True
		print("Training Data")
		predicted = predict(my_tree,train_x_new,modified,max_depth,max_depth)
		print(confusion_matrix(train_y,predicted))
		print(accuracy_score(train_y,predicted))

		print("Testing Data")
		predicted = predict(my_tree,test_x_new,modified,max_depth,max_depth)
		print(confusion_matrix(test_y,predicted))
		print(accuracy_score(test_y,predicted))

		print("Validation Data")
		predicted = predict(my_tree,validation_x_new,modified,max_depth,max_depth)
		print(confusion_matrix(val_y,predicted))
		print(accuracy_score(val_y,predicted))

		# For plotting purpose
		for depth in range(max_depth):
			part1_train_accuracy[depth] = accuracy_score(train_y,predict(my_tree,train_x_new,modified,depth,max_depth))
			part1_test_accuracy[depth] = accuracy_score(test_y,predict(my_tree,test_x_new,modified,depth,max_depth))
			part1_val_accuracy[depth] = accuracy_score(val_y,predict(my_tree,validation_x_new,modified,depth,max_depth))

		fig = plt.figure()
		plt.title("Accuracies vs Depth")
		plt.plot(part1_test_accuracy.keys(),part1_test_accuracy.values(),label = 'Testing')
		plt.plot(part1_val_accuracy.keys(),part1_val_accuracy.values(), label = 'Validation')
		plt.plot(part1_train_accuracy.keys(),part1_train_accuracy.values(), label = 'Training')
		plt.xlabel("Depth")
		plt.ylabel("Accuracies")
		plt.legend()
		# plt.show()
		fig.savefig("Graph_Part3_2"+'.png')

	elif part_num==4:
		# SET OF BEST FEATURE VALUES. To directly use them comment the nested for loop section below
		best_accuracy = -1
		best_depth = 6
		best_min_sample_leaf = 6
		best_min_sample_split = 3

		depth_list = [1,2,3,4,5,6,7,8,9,10,11,12]
		min_sample_leaf_list = [2,3,4,5,6,7,8,9,10]
		min_sample_split_list = [2,3,4,5,6,7,8,9,10]

		for depth in depth_list:
			for min_sample_leaf in min_sample_leaf_list:
				for min_sample_split in min_sample_split_list:
					dec_tree = tree.DecisionTreeClassifier(criterion="entropy",splitter="best",max_depth=depth,min_samples_split=min_sample_split,min_samples_leaf=min_sample_leaf)
					dec_tree = dec_tree.fit(train_x,train_y)
					acc = accuracy_score(val_y,np.array(dec_tree.predict(val_x),dtype=int))
					if acc>best_accuracy:
						print(str(acc))
						best_accuracy = acc
						best_depth = depth
						best_min_sample_split = min_sample_split
						best_min_sample_leaf = min_sample_leaf

		print("best_depth -> " + str(best_depth))
		print("best_min_sample_leaf -> " + str(best_min_sample_leaf))
		print("best_min_sample_split -> " + str(best_min_sample_split))

		variation_with_sample_split = {}
		for i in min_sample_split_list:
			dec_tree = tree.DecisionTreeClassifier(criterion="entropy",splitter="best",max_depth=best_depth,min_samples_split=i,min_samples_leaf=best_min_sample_leaf)
			dec_tree = dec_tree.fit(train_x,train_y)
			acc = accuracy_score(val_y,np.array(dec_tree.predict(val_x),dtype=int))
			variation_with_sample_split[i] = acc

		variation_with_depth = {}
		for i in depth_list:
			dec_tree = tree.DecisionTreeClassifier(criterion="entropy",splitter="best",max_depth=i,min_samples_split=best_min_sample_split,min_samples_leaf=best_min_sample_leaf)
			dec_tree = dec_tree.fit(train_x,train_y)
			acc = accuracy_score(val_y,np.array(dec_tree.predict(val_x),dtype=int))
			variation_with_depth[i] = acc

		variation_with_sample_leaf = {}
		for i in min_sample_leaf_list:
			dec_tree = tree.DecisionTreeClassifier(criterion="entropy",splitter="best",max_depth=best_depth,min_samples_split=best_min_sample_split,min_samples_leaf=i)
			dec_tree = dec_tree.fit(train_x,train_y)
			acc = accuracy_score(val_y,np.array(dec_tree.predict(val_x),dtype=int))
			variation_with_sample_leaf[i] = acc

		fig = plt.figure()
		plt.title("Accuracy vs Depth")
		plt.plot(variation_with_depth.keys(),variation_with_depth.values())
		plt.xlabel("Depth")
		plt.ylabel("Accuracies")
		plt.legend()
		# plt.show()
		fig.savefig("Accuracy_with_depth"+'.png')

		fig = plt.figure()
		plt.title("Accuracy vs min_sample_split")
		plt.plot(variation_with_sample_split.keys(),variation_with_sample_split.values())
		plt.xlabel("min_sample_split")
		plt.ylabel("Accuracy")
		# plt.show()
		fig.savefig("Accuracy_with_min_sample_split"+'.png')

		fig = plt.figure()
		plt.title("Accuracy vs min_sample_leaf")
		plt.plot(variation_with_sample_leaf.keys(),variation_with_sample_leaf.values())
		plt.xlabel("min_sample_leaf")
		plt.ylabel("Accuracy")
		# plt.show()
		fig.savefig("Accuracy_with_min_sample_leaf"+'.png')


		dec_tree = tree.DecisionTreeClassifier(criterion="entropy",splitter="best",max_depth=best_depth,min_samples_split=best_min_sample_split,min_samples_leaf=best_min_sample_leaf)
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
		num_categories = [2,4,3,12,12,12,12,12,12]
		negative_cols = [5,6,7,8,9,10]
		
		train_x_new = ohe_func(train_x,categorical_features,num_categories,negative_cols)
		test_x_new = ohe_func(test_x,categorical_features,num_categories,negative_cols)
		val_x_new = ohe_func(val_x,categorical_features,num_categories,negative_cols)

		criteria = "entropy"
		depth = 6
		min_sample_leaf = 6
		min_sample_split = 3
		dec_tree = tree.DecisionTreeClassifier(criterion=criteria,splitter="best",max_depth=depth,min_samples_split=min_sample_split,min_samples_leaf=min_sample_leaf)
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
		num_categories = [2,4,3,12,12,12,12,12,12]
		negative_cols = [5,6,7,8,9,10]
		
		train_x_new = ohe_func(train_x,categorical_features,num_categories,negative_cols)
		test_x_new = ohe_func(test_x,categorical_features,num_categories,negative_cols)
		val_x_new = ohe_func(val_x,categorical_features,num_categories,negative_cols)

		# SET OF BEST FEATURE VALUES. To directly use them comment the nested for loop section below
		best_num_estimators = 14
		best_bootstrap = True
		best_num_features = 52
		best_accuracy = -1

		num_estimators_list = [4,5,6,7,8,9,10,11,12,13,14,15]
		bootstrap_list = [True,False]
		max_featues_list = []
		for i in range(1,train_x_new.shape[1]):
			max_featues_list.append(i)

		for ne in num_estimators_list:
			for bs in bootstrap_list:
				for mf in max_featues_list:
					rf = RandomForestClassifier(n_estimators=ne,criterion="entropy",bootstrap=bs,max_features=mf)
					rf.fit(train_x_new,np.ravel(train_y))
					acc = accuracy_score(val_y,np.array(rf.predict(val_x_new),dtype=int))
					if acc>best_accuracy:
						print(acc)
						best_accuracy = acc
						best_num_estimators = ne
						best_bootstrap = bs
						best_num_features = mf

		print("best num estimators -> " + str(best_num_estimators))
		print("bootstrap value -> " + str(best_bootstrap))
		print("max_features -> " + str(best_num_features))

		variation_with_n_estimators = {}
		for ne in num_estimators_list:
			rf = RandomForestClassifier(n_estimators=ne,criterion="entropy",bootstrap=best_bootstrap,max_features=best_num_features)
			rf.fit(train_x_new,np.ravel(train_y))
			acc = accuracy_score(val_y,np.array(rf.predict(val_x_new),dtype=int))
			variation_with_n_estimators[ne] = acc

		variation_with_bootstrap = {}
		for i in range(len(bootstrap_list)):
			rf = RandomForestClassifier(n_estimators=best_num_estimators,criterion="entropy",bootstrap=bootstrap_list[1-i],max_features=best_num_features)
			rf.fit(train_x_new,np.ravel(train_y))
			acc = accuracy_score(val_y,np.array(rf.predict(val_x_new),dtype=int))
			variation_with_bootstrap[i] = acc

		variation_with_max_features = {}
		for mf in max_featues_list:
			rf = RandomForestClassifier(n_estimators=best_num_estimators,criterion="entropy",bootstrap=best_bootstrap,max_features=mf)
			rf.fit(train_x_new,np.ravel(train_y))
			acc = accuracy_score(val_y,np.array(rf.predict(val_x_new),dtype=int))
			variation_with_max_features[mf] = acc

		fig = plt.figure()
		plt.title("Accuracy vs n_estimators")
		plt.plot(variation_with_n_estimators.keys(),variation_with_n_estimators.values())
		plt.xlabel("n_estimators")
		plt.ylabel("Accuracies")
		plt.legend()
		# plt.show()
		fig.savefig("Accuracy_with_n_estimators"+'.png')

		fig = plt.figure()
		plt.title("Accuracy vs Bootstrap")
		plt.plot(variation_with_bootstrap.keys(),variation_with_bootstrap.values())
		plt.xlabel("Bootstrap")
		plt.ylabel("Accuracies")
		plt.legend()
		# plt.show()
		fig.savefig("Accuracy_with_bootstrap"+'.png')

		fig = plt.figure()
		plt.title("Accuracy vs max_features")
		plt.plot(variation_with_max_features.keys(),variation_with_max_features.values())
		plt.xlabel("Max Features")
		plt.ylabel("Accuracies")
		plt.legend()
		# plt.show()
		fig.savefig("Accuracy_with_max_features"+'.png')

		rf = RandomForestClassifier(n_estimators=best_num_estimators,criterion="entropy",bootstrap=best_bootstrap,max_features=best_num_features)
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