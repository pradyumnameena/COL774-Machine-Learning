import os
import sys
import cv2
import time
import torch
import shutil
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class Neural_Net(nn.Module):
	def __init__(self):
		super(Neural_Net,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=0)
		self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
		self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=0)
		self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
		self.fc1 = nn.Linear(37440,2048)
		self.fc2 = nn.Linear(2048,2)

	def forward(self,x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = x.view(-1,9*64*65)
		x = self.fc1(x)
		x = self.fc2(x)
		x = F.softmax(x,dim=1)
		return(x)

class BreakOutLoader(Dataset):
	def __init__(self,folder_num,transform=None):
		self.path = "./dataset/train_dataset/"+str(folder_num).zfill(8)+"/new_data/"
		self.reward = np.ravel(np.array(pd.read_csv("./dataset/train_dataset/"+str(folder_num).zfill(8)+"/new_data/new_rewards.csv"))).tolist()
		self.transform = transform

	def __len__(self):
		return len(os.listdir(self.path))-2

	def __getitem__(self,idx):
		image = torch.from_numpy(cv2.imread(self.path+str(idx)+".png").reshape(3,1050,160)).float()
		a = torch.from_numpy(np.eye(2)[self.reward[idx]]).float()
		sample = {'image':image,'label':a}
		return sample
		
def modify_dataset(datapath):
	folder_list = os.listdir(datapath+"/")[2:]
	for folder in folder_list:
		if os.path.exists(datapath+"/"+folder+"/new_data"):
			print(folder + "erased")
			shutil.rmtree(datapath+"/"+folder+"/new_data")
	time1 = time.clock()
	folder_list = os.listdir(datapath+"/")[2:]
	for folder in folder_list:
		os.mkdir(datapath+"/"+folder+"/new_data/")
		counter = 0
		all_files = os.listdir(datapath+"/"+folder+"/")
		rew_file = np.array(pd.read_csv(datapath+"/"+folder+"/rew.csv",header=None,dtype=int))
		png_files = []
		reward_list = []
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		for last_frame in range(6,len(png_files)-2):
			frame_num_list = [x for x in range(last_frame-6,last_frame+1)]
			curr_data = []
			for frame_num in frame_num_list:
				curr_data.append(cv2.imread(datapath+"/"+folder+"/"+png_files[frame_num]))
			for drop1 in range(5):
				for drop2 in range(drop1+1,6):
					new_list = [y for y in [0,1,2,3,4,5,6] if y not in [drop1,drop2]]
					new_data = np.array(np.zeros((1,160,3),dtype=int))
					for idx in new_list:
						new_data = np.concatenate((new_data,curr_data[idx]),axis=0)
					cv2.imwrite(datapath+"/"+folder+"/new_data/"+str(counter)+".png",new_data[1:,:,:])
					del(new_list)
					del(new_data)
					del(idx)
					reward_list.append(rew_file[last_frame+1])
					counter+=1
					# break
				# break
			del(drop1)
			del(drop2)
			del(curr_data)
			del(frame_num)
			del(frame_num_list)
			# break
		pd.DataFrame(np.array(reward_list)).to_csv(datapath+"/"+folder+"/new_data/new_rewards.csv",header=None,index=None)
		del(last_frame)
		del(all_files)
		del(png_files)
		del(reward_list)
		del(counter)
		break
	time2 = time.clock()
	print("Data Modification -> " + str(time2-time1))

def main():
	# modify_dataset("train_dataset")
	my_net = Neural_Net()
	criteria = nn.BCELoss()
	optimizer = optim.Adam(my_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	max_epochs = 1
	folder_list = os.listdir("./dataset/train_dataset/")[2:]
	for epoch in range(max_epochs):
		for idx in range(1,501,1):
			trainloader = BreakOutLoader(folder_num=idx)
			dataloader = DataLoader(trainloader,batch_size=10,num_workers=4,shuffle=True)
			running_loss = 0.0
			for i,data in enumerate(dataloader,0):
				inputs,labels = data
				inputs = Variable(data['image'])
				labels = Variable(data['label'])
				optimizer.zero_grad()
				outputs = my_net(inputs)
				loss = criteria(outputs,labels)
				loss.backward()
				optimizer.step()
				running_loss+=loss.item()

if __name__ == '__main__':
	main()