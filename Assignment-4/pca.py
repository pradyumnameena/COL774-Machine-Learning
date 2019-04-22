import os
import sys
import cv2
import time
import pickle
import numpy as np
import pandas as pd
sys.path.append('/home/cse/btech/cs1160375/version4/libsvm-3.23/python')
from svmutil import *
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,f1_score

def generate_pca_dataset(datapath):
	time1 = time.clock()
	folders = [
	'00000001', '00000002', '00000003', '00000004', '00000005', '00000006', '00000007', '00000008', '00000009', '00000010',
	'00000011', '00000012', '00000013', '00000014', '00000015', '00000016', '00000017', '00000018', '00000019', '00000020', 
	'00000021', '00000022', '00000023', '00000024', '00000025', '00000026', '00000027', '00000028', '00000029', '00000030', 
	'00000031', '00000032', '00000033', '00000034', '00000035', '00000036', '00000037', '00000038', '00000039', '00000040', 
	'00000041', '00000042', '00000043', '00000044', '00000045', '00000046', '00000047', '00000048', '00000049', '00000050']
	gray_list = []
	for folder in folders:
		all_files = os.listdir(datapath+"/"+folder+"/")
		png_files = []
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		for file in png_files:
			gray_list.append(np.ravel(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+file)[32:196,11:153,:], cv2.COLOR_BGR2GRAY)))
		del(all_files)
		del(png_files)
	a = np.array(gray_list)
	del(gray_list)
	del(folders)
	time2 = time.clock()
	pca_dataset_pickle_file = open("pca_dataset_pickle",'ab')
	pickle.dump(a,pca_dataset_pickle_file)
	pca_dataset_pickle_file.close()
	time3 = time.clock()
	print("Time taken for generating list -> " + str(time2-time1))
	print("Time taken for saving array into pickle file -> "+ str(time3-time2))
	print("Dataset generated!!! Hurray")

def pca_transform_folder_wise(datapath,my_pca):
	time1 = time.clock()
	folder_list = os.listdir(datapath+"/")
	counter = 0
	for folder in folder_list:
		time2 = time.clock()
		print(folder)
		all_files = os.listdir(datapath+"/"+folder+"/")
		png_files = []
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		curr_data = []
		for file in png_files:
			curr_data.append(my_pca.transform(np.ravel(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+file)[22:196,11:153,:], cv2.COLOR_BGR2GRAY)).reshape(1,-1)).tolist())
		del(all_files)
		del(png_files)
		svm_train_x = open(datapath+"/"+folder+"/pca_transformed_list",'ab')
		pickle.dump(curr_data,svm_train_x)
		svm_train_x.close()
		time3 = time.clock()
		print("Time taken to generate and store pickle for array = " + str(time3-time2))
	time3 = time.clock()
	print("Time taken to generate all pickle files = " + str(time3-time1))

def generate_svm_test_data(datapath,my_pca):
	time1 = time.clock()
	folder_list = os.listdir(datapath+"/")
	folder_list.remove("rewards.csv")
	list_for_array = []
	reward_list = np.array(pd.read_csv(datapath + "/rewards.csv",header=None,dtype=int))[:,1].tolist()
	for folder in folder_list:
		all_files = os.listdir(datapath+"/"+folder+"/")
		png_files = []
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		for file in png_files:
			list_for_array.append(my_pca.transform(np.ravel(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+file)[22:196,11:153,:], cv2.COLOR_BGR2GRAY)).reshape(1,-1)).tolist())
		del(all_files)
		del(png_files)
	time2 = time.clock()
	print("Time taken to generate list = " + str(time2-time1))

	svm_val_x = open("svm_val_pickle_x",'ab')
	pickle.dump(list_for_array,svm_val_x)
	svm_val_x.close()
	time3 = time.clock()
	print("Time taken to generate pickle file for validation x = " + str(time3-time2))

	svm_val_y = open("svm_val_pickle_y",'ab')
	pickle.dump(reward_list,svm_val_y)
	svm_val_y.close()
	time4 = time.clock()
	print("Time taken to generate pickle file for validation y = " + str(time4-time3))

def get_train_data(datapath):
	time1 = time.clock()
	train_x = []
	train_y = []
	folder_list = os.listdir(datapath+"/")
	for folder in folder_list:
		time2 = time.clock()
		reward_array = np.array(pd.read_csv(datapath+folder+"/rew.csv",dtype=int)).tolist()
		pickle_file = open(datapath+"/"+folder+"/pca_transformed_list",'rb')
		curr_data = pickle.load(pickle_file)
		pickle_file.close()
		del(pickle_file)
		for last_frame in range(6,len(curr_data)-3,1):
			frame_num_list = [x for x in range(last_frame-6,last_frame+1)]
			data = []
			for frame_num in frame_num_list:
				data.append(curr_data[frame_num][0])
			if reward_array[last_frame+1]==1:
				for drop1 in range(5):
					for drop2 in range(drop1+1,6):
						new_list = [y for y in [0,1,2,3,4,5,6] if y not in [drop1,drop2]]
						new_data = []
						for idx in new_list:
							new_data.extend(data[idx])
						train_x.append(new_data)
						train_y.append(reward_array[last_frame+1][0])
						del(new_list)
						del(new_data)
						del(idx)
			else:
				for drop1 in range(5):
					for drop2 in range(drop1+1,6):
						new_list = [y for y in [0,1,2,3,4,5,6] if y not in [drop1,drop2]]
						new_data = []
						for idx in new_list:
							new_data.extend(data[idx])
						train_x.append(new_data)
						train_y.append(reward_array[last_frame+1][0])
						del(new_list)
						del(new_data)
						del(idx)
			del(drop1)
			del(drop2)
			del(data)
			del(frame_num)
			del(frame_num_list)
			time3 = time.clock()
		print("Time taken to generate data from "+folder+" -> "+str(time3-time2))
	time2 = time.clock()
	print("Total time taken to generate train data -> " + str(time2-time1))
	return train_x,train_y

def main():
	generate_pca_dataset("train_dataset")
	pca_dataset_pickle_file = open("pca_dataset_pickle",'rb')
	dataset = pickle.load(pca_dataset_pickle_file)
	my_pca = PCA(n_components=50)
	my_pca.fit(dataset)
	pca_transform_folder_wise("train_dataset",my_pca)
	generate_svm_test_data("validation_dataset",my_pca)
	del(my_pca)

	train_x,train_y = get_train_data("train_dataset")
	problem = svm_problem(train_y,train_x)

	svm_test_x = open("svm_val_pickle_x",'rb')
	val_x = pickle.load(svm_test_x)
	svm_test_x.close()
	del(svm_test_x)

	svm_test_y = open("svm_val_pickle_y",'rb')
	val_y = pickle.load(svm_test_y)
	svm_test_y.close()
	del(svm_test_y)

	penalty = 1
	gamma = 0.05
	
	linear_param = svm_parameter("-s 0 -c " + str(penalty) + " -t 0")
	linear_model = svm_train(problem,linear_param)
	linear_pred_lbl, linear_pred_acc, linear_pred_val = svm_predict(train_y,train_x,linear_model)
	print("Linear Model with penalty = " + str(penalty) + ": ")
	print("Train Accuracy -> " + str(accuracy_score(np.array(train_y),linear_pred_lbl)))
	print("Train F1_score -> " + str(f1_score(np.array(train_y),linear_pred_lbl,average="macro")))
	linear_pred_lbl, linear_pred_acc, linear_pred_val = svm_predict(val_y,val_x,linear_model)
	print("Test Accuracy -> " + str(accuracy_score(np.array(val_y),linear_pred_lbl)))
	print("Test F1_score -> " + str(f1_score(np.array(val_y),linear_pred_lbl,average="macro")))
	
	gaussian_param = svm_parameter("-s 0 -c " + str(penalty) + " -t 2 -g " + str(gamma))
	gaussian_model = svm_train(problem,gaussian_param)
	gaussian_pred_lbl, gaussian_pred_acc, gaussian_pred_val = svm_predict(train_y,train_x,gaussian_model)
	print("Gaussian Model with penalty = " + str(penalty) + " and gamma = " + str(gamma))
	print("Train Accuracy -> " + str(accuracy_score(np.array(train_y),gaussian_pred_lbl)))
	print("Train F1_score -> " + str(f1_score(np.array(train_y),gaussian_pred_lbl,average="macro")))
	gaussian_pred_lbl, gaussian_pred_acc, gaussian_pred_val = svm_predict(val_y,val_x,gaussian_model)
	print("Test Accuracy -> " + str(accuracy_score(np.array(val_y),gaussian_pred_lbl)))
	print("Test F1_score -> " + str(f1_score(np.array(val_y),gaussian_pred_lbl,average="macro")))	

if __name__ == '__main__':
	main()