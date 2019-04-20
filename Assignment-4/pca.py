import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
# from svmutil import *

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
		print(folder)
		all_files = os.listdir(datapath+"/"+folder+"/")
		png_files = []
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		for file in png_files:
			gray_list.append(np.ravel(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+file), cv2.COLOR_BGR2GRAY)))
		del(all_files)
		del(png_files)
	a = np.array(gray_list)
	del(gray_list)
	del(folders)
	time2 = time.clock()
	pd.DataFrame(a).to_csv("pca_dataset_50_epsd.csv",header=None,index=None)
	time3 = time.clock()
	print("Time taken for generating list -> " + str(time2-time1))
	print("Time taken for saving into csv file -> "+ str(time3-time2))
	print("Dataset generated!!! Hurray")

def np_pca(dataset):
	time1 = time.clock()
	dataset = np.subtract(dataset,np.mean(dataset,axis=0))
	co_variance = np.cov(dataset,rowvar=False)
	E,V = np.linalg.eigh(co_variance)
	key = np.argsort(E)[::-1][0:50]
	V = V[:, key]
	time2 = time.clock()
	pd.DataFrame(np.array(V)).to_csv("pca_50_principal_components.csv",header=None,index=None)
	time3 = time.clock()
	print("Time taken for generating principal components -> " + str(time2-time1))
	print("Time taken for saving into csv file -> " + str(time3-time2))
	print("numpy PCA complete!!! Hurray")

def generate_svm_dataset(pca_comp_matrix,datapath):
	folder_list = os.listdir(datapath+"/")[2:]
	folder_list = folder_list[:-1]
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
				curr_data.append(np.dot(cv2.imread(datapath+"/"+folder+"/"+png_files[frame_num]),pca_comp_matrix))
			if rew_file[last_frame]==1:
				for drop1 in range(len(frame_num_list)-2):
					for drop2 in range(drop1+1,len(frame_num_list)-1):
						new_list = [y for y in [0,1,2,3,4,5,6] if y not in [drop1,drop2]]
						new_data = np.array(np.zeros((1,160,3),dtype=float))
						for idx in new_list:
							new_data = np.concatenate((new_data,curr_data[idx]),axis=0)
						del(new_list)
						list_for_array.append(np.array(new_data[1:,:,:]))
						reward_list.append(1)
						pros+=1
			else:
				for drop1 in range(len(frame_num_list)-2):
					for drop2 in range(drop1+1,len(frame_num_list)-1):
						new_list = [y for y in [0,1,2,3,4,5,6] if y not in [drop1,drop2]]
						new_data = np.array(np.zeros((1,160,3),dtype=float))
						for idx in new_list:
							new_data = np.concatenate((new_data,curr_data[idx]),axis=0)
						del(new_list)
						list_for_array.append(np.array(new_data[1:,:,:]))
						reward_list.append(0)
						pros+=1
			del(curr_data)
			del(frame_num_list)
		del(all_files)
		del(png_files)
		pros_list.append(pros)
		cons_list.append(cons)
		print("Total pros -> " + str(pros_list[counter]))
		print("Total cons -> " + str(cons_list[counter]))
		counter+=1
	return list_for_array,reward_list

def generate_svm_data(datapath,pca_comp_matrix):
	folder_list = os.listdir(datapath+"/")[2:]
	folder_list = folder_list[:-1]
	list_for_array = []
	reward_list = np.array(pd.read_csv(datapath + "/rewards.csv",header=None,dtype=int))[:,1].tolist()
	for folder in folder_list:
		all_files = os.listdir(datapath+"/"+folder+"/")
		png_files = []
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		for file in png_files:
			curr_data.append(np.dot(cv2.imread(datapath+"/"+folder+"/"+files),pca_comp_matrix))
		del(all_files)
		del(png_files)
	return list_for_array,reward_list

def main():
	generate_pca_dataset("../../train_dataset")
	dataset = pd.read_csv("../../pca_dataset_50_epsd.csv",header=None,dtype=float)
	np_pca(np.array(dataset))
	pca_components = np.asmatrix(np.array(pd.read_csv("../../pca_50_principal_components.csv",header=None)))
	datapath = "../../train_dataset"
	(train_x,train_y) = generate_svm_dataset(pca_components,datapath)

	# loading the validation data
	(val_x,val_y) = generate_svm_data("../../validation_dataset",pca_components)
	del(pca_components)

	problem = svm_problem(train_x,train_y)
	del(train_x)
	del(train_y)
	linear_param = svm_parameter("-s 0 -c 1 -t 0")
	linear_model = svm_train(problem,linear_param)
	linear_pred_lbl, linear_pred_acc, linear_pred_val = svm_predict(val_y,val_x,linear_model)
	gaussian_param = svm_parameter("-s 0 -c " + str(penalty) + " -t 2 -g " + str(gamma))
	gaussian_model = svm_train(problem,gaussian_param)
	gaussian_pred_lbl, gaussian_pred_acc, gaussian_pred_val = svm_predict(val_y,val_x,gaussian_model)

if __name__ == '__main__':
	main()
