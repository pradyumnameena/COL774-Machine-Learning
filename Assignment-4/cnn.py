import os
import torch
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class Neural_Net(nn.Module):
	def __init__(self):
		super(Neural_Net,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=0)
		self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
		self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=0)
		self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
		# no padding in pooling layers
		# with padding in conv layers of 1
		# self.fc1 = nn.Linear(36864,2048)
		# without padding i.e. padding in conv layer = 0
		self.fc1 = nn.Linear(28224,2048)
		self.fc2 = nn.Linear(2048,2)

	def forward(self,x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		# x = x.view(-1,9*64*64)
		x = x.view(-1,7*63*64)
		x = self.fc1(x)
		x = self.fc2(x)
		return(x)

def generate_dataset(datapath):
	folders =  os.listdir(datapath+"/")
	for folder in folders:
		all_files = os.listdir(datapath+"/"+folder+"/")
		csv_file = datapath + "/" + folder + "/rew.csv"
		png_files = []
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		for i in range(len(png_files)/7):
			a = np.array(np.zeros((1,160,3),dtype=int))
			for k in range(7):
				a = np.concatenate((a,imageio.imread(datapath+"/"+folder+"/"+png_files[7*i+k])),axis=0)
			a = a[1:,:,:]
			del(a)

def main():
	datapath = "train_dataset"
	generate_dataset(datapath)
	my_net = Neural_Net()

if __name__ == '__main__':
	main()