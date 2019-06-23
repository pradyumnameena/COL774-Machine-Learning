import os
import sys
import cv2
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score

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
			gray_list.append(np.float32(np.ravel(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+file)[32:196,11:153,:], cv2.COLOR_BGR2GRAY))/255.0))
		del(all_files)
		del(png_files)
	a = np.array(gray_list)
	del(gray_list)
	del(folders)
	time2 = time.clock()
	print("Time taken to generate dataset for PCA -> " + str(time2-time1))
	return a

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
			curr_data.append(my_pca.transform(np.float32(np.ravel(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+file)[32:196,11:153,:], cv2.COLOR_BGR2GRAY)).reshape(1,-1)/255.0)).tolist())
		del(all_files)
		del(png_files)
		svm_train_x = open(datapath+"/"+folder+"/pca_normalized_transform_list",'ab')
		pickle.dump(curr_data,svm_train_x)
		svm_train_x.close()
		time3 = time.clock()
		print("Time taken to generate and store pickle = " + str(time3-time2))
	time3 = time.clock()
	print("Time taken to generate all pickle files folder wise = " + str(time3-time1))

def synthesize_svm_data(datapath,my_pca,name,reward_avail):
	time1 = time.clock()
	folder_list = os.listdir(datapath+"/")
	if reward_avail==1:
		folder_list.remove("rewards.csv")
		reward_list = np.array(pd.read_csv(datapath + "/rewards.csv",header=None,dtype=int))[:,1].tolist()
	list_for_array = []
	for folder in folder_list:
		all_files = os.listdir(datapath+"/"+folder+"/")
		png_files = []
		curr_data = []
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		for file in png_files:
			curr_data.extend(my_pca.transform(np.float32(np.ravel(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+file)[32:196,11:153,:], cv2.COLOR_BGR2GRAY)).reshape(1,-1)/255.0).tolist()))
		list_for_array.append(curr_data)
		del(curr_data)
		del(all_files)
		del(png_files)
	time2 = time.clock()
	print("Time taken to generate list = " + str(time2-time1))

	svm_val_x = open(name+"x",'ab')
	pickle.dump(list_for_array,svm_val_x)
	svm_val_x.close()
	time3 = time.clock()
	print("Time taken to generate pickle file for x = " + str(time3-time2))

	if reward_avail==1:
		svm_val_y = open(name+"y",'ab')
		pickle.dump(reward_list,svm_val_y)
		svm_val_y.close()
		time4 = time.clock()
		print("Time taken to generate pickle file for y = " + str(time4-time3))

def get_train_data(datapath,num_folders):
	time1 = time.clock()
	train_x = []
	train_y = []
	folder_list = os.listdir(datapath+"/")
	for folder_idx in range(num_folders):
		folder = folder_list[folder_idx]
		time2 = time.clock()
		reward_array = np.array(pd.read_csv(datapath+"/"+folder+"/rew.csv",dtype=int)).tolist()
		pickle_file = open(datapath+"/"+folder+"/pca_normalized_transform_list",'rb')
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
						if last_frame%3==0:
							train_x.append(new_data)
							train_y.append(reward_array[last_frame+1][0])
						del(new_list)
						del(new_data)
			else:
				count_max = folder_idx%2
				for count in range(count_max):
					drop1 = 0
					drop2 = 0
					while drop1==drop2:
						drop1 = np.random.randint(6)
						drop2 = np.random.randint(6)
					new_list = [y for y in [0,1,2,3,4,5,6] if y not in [drop1,drop2]]
					new_data = []
					for idx in new_list:
						new_data.extend(data[idx])
					train_x.append(new_data)
					train_y.append(reward_array[last_frame+1][0])
					del(new_list)
					del(new_data)
			del(data)
			del(frame_num)
			del(frame_num_list)
			time3 = time.clock()
		print("Time taken to generate data from "+folder+" -> "+str(time3-time2))
	time2 = time.clock()
	print("Total time taken to generate train data -> " + str(time2-time1))
	return train_x,train_y
	 
def get_test_data(val_or_test,reward_avail):
	path = ""
	val_x = []

	if val_or_test==0:
		path = "./pca_val_normalized"
	else:
		path = "pca_test_normalized"

	# Loading the x file
	pickle_file = open(path+"x",'rb')
	data_dash = pickle.load(pickle_file)
	pickle_file.close()
	for i in range(len(data_dash)):
		curr_data = []
		for j in range(5):
			curr_data.extend(data_dash[i][j])
		val_x.append(curr_data)

	if val_or_test==1:
		return val_x
	else:
		pickle_file = open(path+"y",'rb')
		data_dash = pickle.load(pickle_file)
		pickle_file.close()
		return val_x,data_dash

def main():
	dataset = generate_pca_dataset("../train_dataset")
	pca_dataset_pickle_file = open("pca_dataset_pickle",'rb')
	dataset = pickle.load(pca_dataset_pickle_file)
	my_pca = PCA(n_components=50)
	my_pca.fit(dataset)
	del(dataset)
	pca_transform_folder_wise("../train_dataset",my_pca)
	synthesize_svm_data("../validation_dataset",my_pca,"pca_val_normalized",1)
	synthesize_svm_data("../test_dataset",my_pca,"pca_test_normalized",0)
	del(my_pca)
	
	time1 = time.clock()
	num_folders = 500
	train_x,train_y = get_train_data("../train_dataset",num_folders)
	time2 = time.clock()
	print("Training data read in " + str(time2-time1))
	
	val_x,val_y = get_test_data(0,1)
	test_x = get_test_data(1,0)
	penalty = 5
	gamma = "auto"

	print("Val and test data read")	
	
	time3 = time.clock()	
	linear_svm = SVC(C=penalty,kernel='linear',max_iter=4000)
	linear_svm.fit(train_x,train_y)
	time4 = time.clock()
	print("Linear trained in " + str(time4-time3))

	time5 = time.clock()
	gaussian_svm = SVC(C=penalty,kernel='rbf',gamma=gamma,max_iter=4000)
	gaussian_svm.fit(train_x,train_y)
	time6 = time.clock()
	print("Gaussian trained in " + str(time6-time5))

	linear_svm_pickle = open("svm_linear_trained",'ab')
	pickle.dump(linear_svm,linear_svm_pickle)
	linear_svm_pickle.close()

	gaussian_svm_pickle = open("gaussian_svm_trained",'ab')
	pickle.dump(gaussian_svm,gaussian_svm_pickle)
	gaussian_svm_pickle.close()

	linear_svm_pickle = open("svm_linear_trained",'rb')
	linear_svm = pickle.load(linear_svm_pickle)
	linear_svm_pickle.close()

	gaussian_svm_pickle = open("gaussian_svm_trained",'rb')
	gaussian_svm = pickle.load(gaussian_svm_pickle)
	gaussian_svm_pickle.close()
	
	linear_pred_lbl = linear_svm.predict(train_x)
	print("Linear Model with penalty = " + str(penalty) + ": ")
	print("Train Accuracy -> " + str(accuracy_score(np.array(train_y),linear_pred_lbl)))
	print("Train F1_score -> " + str(f1_score(np.array(train_y),linear_pred_lbl,average="micro")))
	print(confusion_matrix(np.array(train_y),linear_pred_lbl))
	print(f1_score(np.array(train_y),linear_pred_lbl))
	
	linear_pred_lbl = linear_svm.predict(val_x)	
	print("Test Accuracy -> " + str(accuracy_score(np.array(val_y),linear_pred_lbl)))
	print("Test F1_score -> " + str(f1_score(np.array(val_y),linear_pred_lbl,average="micro")))
	print(confusion_matrix(np.array(val_y),linear_pred_lbl))
	print(f1_score(np.array(val_y),linear_pred_lbl))
	# time5 = time.clock()
	
	gaussian_pred_lbl = gaussian_svm.predict(train_x)
	print("Gaussian Model with penalty = " + str(penalty) + " and gamma = " + str(gamma))
	print("Train Accuracy -> " + str(accuracy_score(np.array(train_y),gaussian_pred_lbl)))
	print("Train F1_score -> " + str(f1_score(np.array(train_y),gaussian_pred_lbl,average="micro")))
	print(confusion_matrix(np.array(train_y),gaussian_pred_lbl))
	print(f1_score(np.array(train_y),gaussian_pred_lbl))
	
	gaussian_pred_lbl = gaussian_svm.predict(val_x)
	print("Test Accuracy -> " + str(accuracy_score(np.array(val_y),gaussian_pred_lbl)))
	print("Test F1_score -> " + str(f1_score(np.array(val_y),gaussian_pred_lbl,average="micro")))	
	print(confusion_matrix(np.array(val_y),gaussian_pred_lbl))
	print(f1_score(np.array(val_y),gaussian_pred_lbl))	
	
	columns = ["Prediction"]
	linear_test_pred = np.ravel(linear_svm.predict(test_x)).tolist()
	gaussian_test_pred = np.ravel(gaussian_svm.predict(test_x)).tolist()
	pd.DataFrame(np.array(linear_test_pred)).to_csv("linear_svm_prediction.csv",header=columns,index=True)
	pd.DataFrame(np.array(gaussian_test_pred)).to_csv("gaussian_svm_prediction.csv",header=columns,index=True)

if __name__ == '__main__':
	main()
