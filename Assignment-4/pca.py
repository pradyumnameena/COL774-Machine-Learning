import os
import sys
import cv2
import time
import pickle
import numpy as np
import pandas as pd
from svmutil import *
from sklearn.decomposition import PCA

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
			gray_list.append(np.ravel(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+file)[32:196,11:150,:], cv2.COLOR_BGR2GRAY)))
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

def generate_svm_dataset(my_pca,datapath):
	time1 = time.clock()
	folder_list = os.listdir(datapath+"/")
	list_for_array = []
	reward_list = []
	counter = 0
	pros_list = []
	cons_list = []
	for folder in folder_list:
		print(folder)
		pros = 0
		cons = 0
		all_files = os.listdir(datapath+"/"+folder+"/")
		rew_file = np.array(pd.read_csv(datapath+"/"+folder+"/rew.csv",header=None,dtype=int))
		png_files = []
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		for last_frame in range(6,len(png_files)-1):
			frame_num_list = [x for x in range(last_frame-6,last_frame+1)]
			curr_data = []
			for frame_num in frame_num_list:
				curr_data.append(my_pca.transform(np.ravel(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+file)[32:196,11:150,:], cv2.COLOR_BGR2GRAY)).reshape(1,-1)).tolist())
			if rew_file[last_frame+1]==1:
				for drop1 in range(len(frame_num_list)-2):
					for drop2 in range(drop1+1,len(frame_num_list)-1):
						new_list = [y for y in [0,1,2,3,4,5,6] if y not in [drop1,drop2]]
						new_data = []
						for idx in new_list:
							new_data.extend(curr_data[idx])
						list_for_array.append(new_data)
						reward_list.append(1)
						pros+=1
						del(idx)
						del(new_data)
						del(new_list)
			else:
				for drop1 in range(len(frame_num_list)-2):
					for drop2 in range(drop1+1,len(frame_num_list)-1):
						new_list = [y for y in [0,1,2,3,4,5,6] if y not in [drop1,drop2]]
						new_data = []
						for idx in new_list:
							new_data.extend(curr_data[idx])
						list_for_array.append(new_data)
						reward_list.append(0)
						cons+=1
						del(idx)
						del(new_data)
						del(new_list)
			del(curr_data)
			del(frame_num_list)
		del(all_files)
		del(png_files)
		pros_list.append(pros)
		cons_list.append(cons)
		print("Total pros -> " + str(pros_list[counter]))
		print("Total cons -> " + str(cons_list[counter]))
		counter+=1
	time2 = time.clock()
	print("Time taken to generate list = " + str(time2-time1))
	svm_train_x = open("svm_training_data_pickle_x",'ab')
	pickle.dump(np.array(list_for_array),svm_train_x)
	svm_train_x.close()
	time3 = time.clock()
	print("Time taken to generate pickle file for x = " + str(time3-time2))

	svm_train_y = open("svm_training_data_pickle_y",'ab')
	pickle.dump(np.array(reward_list),svm_train_y)
	svm_train_y.close()
	time4 = time.clock()
	print("Time taken to generate pickle file for y = " + str(time4-time3))

def generate_svm_test_data(datapath,my_pca):
	time1 = time.clock()
	folder_list = os.listdir(datapath+"/")[2:]
	list_for_array = []
	reward_list = np.array(pd.read_csv(datapath + "/rewards.csv",header=None,dtype=int))[:,1].tolist()
	for folder in folder_list:
		all_files = os.listdir(datapath+"/"+folder+"/")
		png_files = []
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		for file in png_files:
			list_for_array.append(my_pca.transform(np.ravel(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+file)[32:196,11:150,:], cv2.COLOR_BGR2GRAY)).reshape(1,-1)).tolist())
		del(all_files)
		del(png_files)
	time2 = time.clock()
	print("Time taken to generate list = " + str(time2-time1))

	svm_val_x = open("svm_val_data_pickle_x",'ab')
	pickle.dump(np.array(list_for_array),svm_val_x)
	svm_val_x.close()
	time3 = time.clock()
	print("Time taken to generate pickle file for validation x = " + str(time3-time2))

	svm_val_y = open("svm_val_data_pickle_y",'ab')
	pickle.dump(np.array(reward_list),svm_val_y)
	svm_val_y.close()
	time4 = time.clock()
	print("Time taken to generate pickle file for validation y = " + str(time4-time3))

def main():
	generate_pca_dataset("../../train_dataset")
	pca_dataset_pickle_file = open("pca_dataset_pickle",'rb')
	dataset = pickle.load(pca_dataset_pickle_file)
	
	my_pca = PCA(n_components=50)
	my_pca.fit(dataset)
	generate_svm_dataset("../../train_dataset",my_pca)
	generate_svm_test_data("../../../validation_dataset",my_pca)
	
	svm_training_data_pickle_x = open("svm_training_data_pickle_x",'rb')
	train_x = pickle.load(svm_training_data_pickle_x)
	svm_training_data_pickle_x.close()
	
	svm_training_data_pickle_y = open("svm_training_data_pickle_y",'rb')
	train_y = pickle.load(svm_training_data_pickle_y)
	svm_training_data_pickle_y.close()
	
	del(my_pca)
	problem = svm_problem(train_x,train_y)
	del(train_x)
	del(train_y)

	penalty = 1
	gamma = 0.05
	linear_param = svm_parameter("-s 0 -c " + str(penalty) + " -t 0")
	linear_model = svm_train(problem,linear_param)
	linear_pred_lbl, linear_pred_acc, linear_pred_val = svm_predict(val_y,val_x,linear_model)
	gaussian_param = svm_parameter("-s 0 -c " + str(penalty) + " -t 2 -g " + str(gamma))
	gaussian_model = svm_train(problem,gaussian_param)
	gaussian_pred_lbl, gaussian_pred_acc, gaussian_pred_val = svm_predict(val_y,val_x,gaussian_model)

if __name__ == '__main__':
	main()