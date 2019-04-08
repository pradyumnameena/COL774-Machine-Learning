import sys
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class tree_Node:
	def __init__(self,datapoints_indices,parent,val=-1,childs = [],num_nodes=1,feature_index=-1,answer=0):
		self.indices = datapoints_indices
		self.childs = childs
		self.parent = parent
		self.value = val
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

def my_entropy(datapoints_indices,data_y):
	new_list = np.ravel(data_y[datapoints_indices])
	bin_count = np.divide(np.bincount(new_list),1.0*len(datapoints_indices))
	bin_count = np.add(bin_count,np.multiply(np.ones(bin_count.shape,dtype=int),bin_count==0))
	return -1*np.sum(np.multiply(bin_count,np.divide(np.log(bin_count),np.log(2))))

def split_parent_correct(feature_index,datapoints_indices,data_x):
	new_data = data_x[datapoints_indices,:]
	dicti = {}
	for i in datapoints_indices:
		for a in np.unique(np.ravel(new_data[:,feature_index])):
			if data_x[i,feature_index]==a:
				if a  not in dicti:
					dicti[a] = []
				dicti[a].append(i)
	return dicti

def split_parent(feature_index,datapoints_indices,data_x):
	new_data2 = data_x[datapoints_indices,:][:,feature_index]
	dicti = {c: [datapoints_indices[x] for x in np.where(new_data2==c)[0]] for c in np.unique(np.ravel(new_data2))}
	return dicti
	
def info_gain(feature_index,datapoints_indices,data_x,data_y):
	dicti = split_parent(feature_index,datapoints_indices,data_x)
	ig = 0.0
	for prob in dicti.keys():
		ig -= (np.divide(1.0*len(dicti[prob]),len(datapoints_indices)) * my_entropy(dicti[prob],data_y))
	return ig

def best_feature(datapoints_indices,data_x,data_y):
	max_gain = -1
	best_feature = -1
	parent_entropy = my_entropy(datapoints_indices,data_y)
	for i in range(data_x.shape[1]):
		ig = parent_entropy + info_gain(i,datapoints_indices,data_x,data_y)
		if ig>max_gain:
			max_gain = ig
			best_feature = i
	return best_feature,max_gain

def print_tree(tree):
	if tree.parent==None:
		print("Root Node. Feature Used -> " + str(tree.feature_index))
		new_list = []
		print(len(tree.childs))
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

def get_class(tree,data_point):
	if tree.feature_index==-1:
		return tree.answer
	else:
		for i in range(len(tree.childs)):
			if data_point[tree.feature_index]==tree.childs[i].value:
				return get_class(tree.childs[i],data_point)
		return tree.answer

def predict(tree,test_x):
	predicted = np.asmatrix(np.zeros((test_x.shape[0],1),dtype=int))
	for j in range(test_x.shape[0]):
		predicted[j,0] = get_class(tree,np.ravel(test_x[j,:]))
	return predicted

def grow_tree(train_x,train_y,datapoints_indices,parent=None):
	new_list = np.ravel(train_y[datapoints_indices])
	
	# PURE LEAF NODE || NO BEST FEATURE AVAILABLE
	if len(set(new_list)) <= 1:
		return tree_Node(datapoints_indices,parent,answer=np.argmax(np.bincount(new_list)),feature_index=-1)

	bf, max_gain = best_feature(datapoints_indices,train_x,train_y)
	if max_gain >= 0:
		dicti = split_parent(bf,datapoints_indices,train_x)
		node = tree_Node(datapoints_indices,parent,-1,[],feature_index=bf,answer=np.argmax(np.bincount(new_list)))

		if len(dicti)==1:
			return node
		for val in dicti.keys():
			child = grow_tree(train_x,train_y,dicti[val],parent=node)
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
		print("Total Nodes = " + str(my_tree.num_nodes))

		print("Training Data")
		predicted = predict(my_tree,train_x_new)
		print(confusion_matrix(train_y,predicted))
		print(accuracy_score(train_y,predicted))

		print("Testing Data")
		predicted = predict(my_tree,test_x_new)
		print(confusion_matrix(test_y,predicted))
		print(accuracy_score(test_y,predicted))

		print("Validation Data")
		predicted = predict(my_tree,validation_x_new)
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
		
		# DO PRUNING HERE

		# print_tree(tree)
		print("Total Nodes = " + str(my_tree.num_nodes))

		print("Training Data")
		predicted = predict(my_tree,train_x_new)
		print(confusion_matrix(train_y,predicted))
		print(accuracy_score(train_y,predicted))

		print("Testing Data")
		predicted = predict(my_tree,test_x_new)
		print(confusion_matrix(test_y,predicted))
		print(accuracy_score(test_y,predicted))

		print("Validation Data")
		predicted = predict(my_tree,validation_x_new)
		print(confusion_matrix(val_y,predicted))
		print(accuracy_score(val_y,predicted))

	elif part_num==3:
		print(part_num)

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