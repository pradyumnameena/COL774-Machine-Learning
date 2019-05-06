import os
import sys
import cv2
import time
import torch
import shutil
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

class Neural_Net(nn.Module):
	def __init__(self):
		super(Neural_Net,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=2,padding=0)
		self.bn1 = nn.BatchNorm2d(32)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
		self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=0)
		self.bn2 = nn.BatchNorm2d(64)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
		self.fc1 = nn.Linear(7680,4096)
		# self.dropout = nn.Dropout(p=0.5)
		# self.fc2 = nn.Linear(4096,1024)
		# self.dropout = nn.Dropout(p=0.4)
		self.fc3 = nn.Linear(4096,2)

	def forward(self,x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu2(x)
		x = self.pool2(x)
		x = x.view(-1,7680)
		x = self.fc1(x)
		# x = self.fc2(x)
		x = self.fc3(x)
		x = F.softmax(x,dim=1)
		return(x)

class BOLD(Dataset):
	def __init__(self,path,csv_avail,crop_version,transform=None):
		self.path = path
		self.reward = [0]*len(os.listdir(self.path))
		if csv_avail==1:
			self.reward = np.ravel(np.array(pd.read_csv(path+"/new_rewards.csv",dtype=int))).tolist()
		self.transform = transform
		self.crop_version = crop_version

	def __len__(self):
		return len(os.listdir(self.path))-4

	def __getitem__(self,idx):
		if self.crop_version==1:
			image_array = torch.from_numpy(cv2.imread(self.path+"/"+str(idx)+".png",0).reshape(1,250,140)).float()
		else:
			image_array = torch.from_numpy(cv2.imread(self.path+"/"+str(idx)+".png",0).reshape(1,820,140)).float()
		image_label = torch.from_numpy(np.eye(2)[self.reward[idx]]).float()
		return {"image":image_array,"label":image_label}

def delete_modification(datapath,crop_version):
	ans = []
	count = 0
	reqd_folder = "cropv1" if crop_version==1 else "cropv2"
	folder_list = os.listdir(datapath+"/")
	for folder in folder_list:
		if os.path.exists(datapath+"/"+folder+"/"+reqd_folder):
			shutil.rmtree(datapath+"/"+folder+"/"+reqd_folder)
			ans.append(folder)
			count+=1
	print(str(count) + " folders")
	print(ans)

def count_modification(datapath,crop_version):
	ans = []
	count = 0
	reqd_folder = "cropv1" if crop_version==1 else "cropv2"
	folder_list = os.listdir(datapath+"/")
	for folder in folder_list:
		if os.path.exists(datapath+"/"+folder+"/"+reqd_folder):
			ans.append(folder)
			count+=1
	print(str(count) + " folders")
	print(ans)

def modify_train_dataset_cropV1(datapath):
	time1 = time.clock()
	folder_list = os.listdir(datapath+"/")
	if '.bash_history' in folder_list:	
		folder_list.remove('.bash_history')
	if '.DS_Store' in folder_list:
		folder_list.remove('.DS_Store')
	for folder_idx in range(1,100,1):
		folder = str(folder_idx).zfill(8)
		os.mkdir(datapath+"/"+folder+"/cropv1/")
		counter = 0
		all_files = os.listdir(datapath+"/"+folder+"/")
		rew_file = np.ravel(np.array(pd.read_csv(datapath+"/"+folder+"/rew.csv",header=None,dtype=int))).tolist()
		png_files = []
		reward_list = []
		num_pos = 0
		num_neg = 0
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		for last_frame in range(6,len(png_files)-2):
			frame_num_list = [x for x in range(last_frame-6,last_frame+1)]
			curr_data = []
			for frame_num in frame_num_list:
				curr_data.append(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+png_files[frame_num])[51:101,10:150,:], cv2.COLOR_BGR2GRAY))
			for drop1 in range(5):
				for drop2 in range(drop1+1,6):
					new_list = [y for y in [0,1,2,3,4,5,6] if y not in [drop1,drop2]]
					new_data = np.array(np.zeros((1,140),dtype=int))
					for idx in new_list:
						new_data = np.concatenate((curr_data[idx],new_data),axis=0)
					cv2.imwrite(datapath+"/"+folder+"/cropv1/"+str(counter)+".png",new_data[:-1,:])
					counter+=1
					del(new_list)
					del(new_data)
					reward_list.append(rew_file[last_frame+1])
			if rew_file[last_frame+1]==1:
				num_pos+=15
			else:
				num_neg+=15
			del(curr_data)
			del(frame_num_list)
		pd.DataFrame(np.array(reward_list)).to_csv(datapath+"/"+folder+"/cropv1/new_rewards.csv",header=None,index=None)
		del(all_files)
		del(png_files)
		del(reward_list)
		print(folder+" -> "+ str(num_pos) + "," + str(num_neg))
	time2 = time.clock()
	print("Data Modification Cropping V1-> " + str(time2-time1))

def modify_train_dataset_cropV2(datapath):
	time1 = time.clock()
	folder_list = os.listdir(datapath+"/")
	folder_list.remove('.bash_history')
	folder_list.remove('.DS_Store')
	for folder in folder_list:
		os.mkdir(datapath+"/"+folder+"/cropv2/")
		counter = 0
		all_files = os.listdir(datapath+"/"+folder+"/")
		rew_file = np.ravel(np.array(pd.read_csv(datapath+"/"+folder+"/rew.csv",header=None,dtype=int))).tolist()
		png_files = []
		reward_list = []
		num_pos = 0
		num_neg = 0
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		for last_frame in range(6,len(png_files)-2):
			frame_num_list = [x for x in range(last_frame-6,last_frame+1)]
			curr_data = []
			for frame_num in frame_num_list:
				curr_data.append(cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+png_files[frame_num])[32:196,10:150,:], cv2.COLOR_BGR2GRAY))
			for drop1 in range(5):
				for drop2 in range(drop1+1,6):
					new_list = [y for y in [0,1,2,3,4,5,6] if y not in [drop1,drop2]]
					new_data = np.array(np.zeros((1,140),dtype=int))
					for idx in new_list:
						new_data = np.concatenate((curr_data[idx],new_data),axis=0)
					cv2.imwrite(datapath+"/"+folder+"/cropv2/"+str(counter)+".png",new_data[:-1,:])
					counter+=1
					del(new_list)
					del(new_data)
					reward_list.append(rew_file[last_frame+1])
			if rew_file[last_frame+1]==1:
				num_pos+=15
			else:
				num_neg+=15
			del(curr_data)
			del(frame_num_list)
		pd.DataFrame(np.array(reward_list)).to_csv(datapath+"/"+folder+"/cropv2/new_rewards.csv",header=None,index=None)
		del(all_files)
		del(png_files)
		del(reward_list)
		print(folder+" -> "+ str(num_pos) + "," + str(num_neg))
		# break
	time2 = time.clock()
	print("Data Modification Cropping V2-> " + str(time2-time1))

def generate_train_dataset(datapath):
	time1 = time.clock()
	folder_list = os.listdir(datapath+"/")
	if ".bash_history" in folder_list:
		folder_list.remove('.bash_history')
	if ".DS_Store" in folder_list:
		folder_list.remove('.DS_Store')
	for folder_idx in range(1,100,1):
		folder = str(folder_idx+1).zfill(8)
		os.mkdir(datapath+"/"+folder+"/train_data/")
		counter = 0
		all_files = os.listdir(datapath+"/"+folder+"/cropv1/")
		rew_file = np.ravel(np.array(pd.read_csv(datapath+"/"+folder+"/cropv1/new_rewards.csv",header=None,dtype=int))).tolist()
		reward_list = []
		pos_list = [x for x in range(len(all_files)-3) if rew_file[x]==1]
		pos_samples = len(pos_list)
		neg_list = [x for x in range(len(all_files)-3) if rew_file[x]==0]
		num_neg_samples = (int)(1.1*len(pos_list))
		for count in range(num_neg_samples):
			idx = np.random.randint(len(neg_list))
			pos_list.append(idx)
		neg_samples = len(pos_list)-pos_samples
		random.shuffle(pos_list)
		for count in range(len(pos_list)):
			cv2.imwrite(datapath+"/"+folder+"/train_data/"+str(count)+".png",cv2.imread(datapath+"/"+folder+"/cropv1/"+str(pos_list[count])+".png",0))
			reward_list.append(rew_file[pos_list[count]])
		pd.DataFrame(np.array(reward_list)).to_csv(datapath+"/"+folder+"/train_data/new_rewards.csv",header=None,index=None)
		del(reward_list)
		print(folder + " -> " + str(pos_samples) + "," + str(neg_samples))
		# if folder == "00000020":
			# break
	time2 = time.clock()
	print("Training Data Generated -> " + str(time2-time1))

def modify_test_dataset(datapath,crop_version,csv_avail):
	time1 = time.clock()
	reqd_folder = "cropv1"
	not_reqd_folder = "cropv2"
	begin = 51
	end = 101
	if crop_version==2:
		begin=32
		end=196
		reqd_folder="cropv2"
		not_reqd_folder = "cropv1"
	
	folder_list = os.listdir(datapath+"/")
	
	if csv_avail:
		folder_list.remove('rewards.csv')
	if not_reqd_folder in folder_list:
		folder_list.remove(not_reqd_folder)

	if '.DS_Store' in folder_list:
		folder_list.remove('.DS_Store')
	os.mkdir(datapath+"/"+reqd_folder+"/")
	counter = 0
	for folder in folder_list:
		all_files = os.listdir(datapath+"/"+folder+"/")
		png_files = []
		for file in all_files:
			png_files.append(file) if ('.png' in file) else None
		new_data = np.array(np.zeros((1,140),dtype=int))
		for frame_num in [0,1,2,3,4]:
			new_data = np.concatenate((cv2.cvtColor(cv2.imread(datapath+"/"+folder+"/"+png_files[frame_num])[begin:end,10:150,:], cv2.COLOR_BGR2GRAY),new_data),axis=0)
		cv2.imwrite(datapath+"/"+reqd_folder+"/"+str(counter)+".png",new_data[:-1,:])
		counter+=1
		del(all_files)
		del(png_files)
	if csv_avail:
		pd.DataFrame(np.array(pd.read_csv(datapath+"/rewards.csv",dtype=int,header=None))[:,1]).to_csv(datapath+"/"+reqd_folder+"/new_rewards.csv",header=None,index=False)
	time2 = time.clock()
	print("Data Modification Cropping version -> " + str(crop_version) + " takes " + str(time2-time1))

def predict(my_net,datapath,csv_avail,crop_version,name):
	testloader2 = BOLD(path=datapath,csv_avail=csv_avail,crop_version=1)
	testloader3 = DataLoader(testloader2,batch_size=4,num_workers=4)
	predicted_labels = []
	true_labels = []
	with torch.no_grad():
		for data in testloader3:
			images,labels = data
			images = Variable(data['image'])
			labels = Variable(data['label'])
			if torch.cuda.is_available():
				images = Variable(data['image']).cuda()
				labels = Variable(data['label']).cuda()
			outputs = my_net(images).cpu()
			_,predicted = torch.max(outputs.data,1)
			# print("1")
			# print(predicted)
			# print(torch.max(labels.cpu().data,1)[1])
			predicted_labels.extend(predicted.numpy().tolist())
			true_labels.extend(torch.max(labels.cpu().data,1)[1].numpy().tolist())
	print("Predictions complete")
	if csv_avail==1:
		print(confusion_matrix(np.array(true_labels),np.array(predicted_labels)))
		print(accuracy_score(np.array(true_labels),np.array(predicted_labels)))
		print(f1_score(np.array(true_labels),np.array(predicted_labels),average="micro"))
	else:
		pd.DataFrame(np.array(predicted_labels)).to_csv(name,header=["Prediction"],index=True)

def main():
	# Modifying train and test data 
	modify_train_dataset_cropV1("../train_dataset")
	# modify_train_dataset_cropV2("./dataset/train_dataset")
	# modify_test_dataset("../validation_dataset",1,True)
	# modify_test_dataset("./dataset/validation_dataset",2,True)
	# modify_test_dataset("../test_dataset",1,False)
	# modify_test_dataset("./dataset/test_dataset",2,False)
	# count_modification("./dataset/train_dataset",2)
	generate_train_dataset("../train_dataset")

	# my_net = Neural_Net()
	# class_weights = torch.FloatTensor([0.85,0.9])
	if torch.cuda.is_available():
		my_net = my_net.cuda()
		# class_weights = class_weights.cuda()
	# criteria = nn.BCELoss(weight=class_weights)
	criteria = nn.BCELoss()
	# optimizer = optim.Adam(my_net.parameters(),lr=0.0001,betas=(0.9,0.999),eps=1e-8,weight_decay=0)
	optimizer = optim.SGD(my_net.parameters(), lr=0.0001, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)

	folder_list = os.listdir("../train_dataset/")
	begin = 1
	batchsize = 32
	end = len(folder_list)
	end = 500
	max_epochs = 10

	# Training Phase
	for epoch in range(max_epochs):
		time1 = time.clock()
		running_loss = 0.0
		num_points = 0
		for idx in range(begin,end+1,1):
			if os.path.exists("./dataset/train_dataset/"+str(idx).zfill(8)+"/train_data"):
				print(idx)
				train_loader = BOLD("./dataset/train_dataset/"+str(idx).zfill(8)+"/train_data/",1,1)
				dataloader = DataLoader(train_loader,batch_size=batchsize,num_workers=4)
				for i,data in enumerate(dataloader):
					inputs,labels = data
					images = Variable(data['image'])
					labels = Variable(data['label'])
					if torch.cuda.is_available():
						images = Variable(data['image']).cuda()
						labels = Variable(data['label']).cuda()
					optimizer.zero_grad()
					outputs = my_net(images)
					loss = criteria(outputs,labels)
					loss.backward()
					optimizer.step()
					num_points+=outputs.size()[0]
					running_loss+=loss.item()
		print("Epoch: " + str(epoch) + " Loss: " + str(running_loss*1.0/num_points))
		time2 = time.clock()
		predict(my_net,"./dataset/validation_dataset/cropv1/",1,1,"val_prediction_cnn.csv")
		time3 = time.clock()
		print("Time for 1 epoch -> " + str(time2-time1))
		print("Time for 1 epoch+prediction -> " + str(time3-time1))
		for i in range(1,20,1):
			print(i)
			predict(my_net,"./dataset/train_dataset/"+str(i).zfill(8)+"/train_data/",1,1,"a.csv")

	time0 = time.clock()
	torch.save(my_net.state_dict(), "model1")
	time1 = time.clock()
	my_net.load_state_dict(torch.load("model1"))
	time2 = time.clock()
	my_net.eval()
	time3 = time.clock()
	print(str(time1-time0))
	print(str(time2-time1))
	print(str(time3-time2))

	for i in range(1,20,1):
		predict(my_net,"./dataset/train_dataset/"+str(i).zfill(8)+"/train_data/",1,1,"a.csv")

	# predict(my_net,"./dataset/validation_dataset/cropv1/",1,1,"val_prediction_cnn.csv")
	# predict(my_net,"../dataset/test_dataset/cropv1/",0,0,"test_prediction_cnn.csv")

if __name__ == '__main__':
	main()
