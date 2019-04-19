import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

def generate_dataset(datapath):
	folders =  os.listdir(datapath+"/")
	folders = folders[1:51]
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
	print("array generated")
	del(gray_list)
	del(folders)
	pd.DataFrame(a).to_csv("pca_dataset_50_epsd.csv",header=None,index=None)

def main():
	data_start = time.clock()
	generate_dataset("train_dataset")
	data_end = time.clock()
	print(str(data_end-data_start) + " is time taken for data generation")
	
	data_start = time.clock()
	b = pd.read_csv("pca_dataset.csv",header=None)
	data_end = time.clock()
	print(str(data_end-data_start) + " is time taken for data loading")
	
	print(b)
	print(b.shape)
	print(np.unique(np.ravel(np.array(b))))
	# print("Main")

if __name__ == '__main__':
	main()