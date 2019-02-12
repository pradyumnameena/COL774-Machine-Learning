import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from decimal import Decimal

def read_params():
  x_data = pd.read_csv("q4x.dat",header=None,sep="\s+",dtype=float)
  y_data = pd.read_csv("q4y.dat",header=None,sep="\s+")
  x = np.asmatrix(np.array(x_data))
  y = np.asmatrix(np.array(y_data))
  x = normalize(x)
  y = modify(y)
  return (x,y)

def normalize(mat):
  mean = np.mean(mat,axis=0)
  var = np.var(mat,axis=0)
  for i in range (len(mat)):
  	for j in range (mat.shape[1]):
  		mat[i,j] = Decimal(mat[i,j]) - Decimal(mean[0,j])
  		mat[i,j] = Decimal(mat[i,j])/Decimal(var[0,j])
  z = np.zeros((mat.shape[0],1),dtype=float)
  z+=1
  z = np.hstack((z,mat))
  return z

def modify(y):
  z = np.asmatrix(np.zeros((y.shape[0],y.shape[1]),dtype=int,order='F'))
  for i in range(len(y)):
    if(y[i,0]=='Alaska'):
      z[i,0] = 1
    else:
      z[i,0] = 0
  return z

def compute_mean(x,y,num_classes):
  count = np.asmatrix(np.zeros((num_classes,1),dtype=int,order='F'))
  summation = np.asmatrix(np.zeros((num_classes,x.shape[1]),dtype=float,order='F'))
  for i in range(len(x)):
    count[y[i],0]+=1
    for j in range(x.shape[1]):
      summation[y[i],j]+=x[i,j]
  for i in range(num_classes):
    for j in range(x.shape[1]):
      summation[i,j]/=count[i,0]
  return summation

def compute_var(x,y,num_classes):
  count = np.asmatrix(np.zeros((num_classes,1),dtype=int,order='F'))
  summation = np.asmatrix(np.zeros((num_classes,x.shape[1]),dtype=float,order='F'))
  for i in range(len(x)):
    count[y[i],0]+=1
    for j in range(x.shape[1]):
      summation[y[i],j]+=x[i,j]
  for i in range(num_classes):
    for j in range(x.shape[1]):
      summation[i,j]/=count[i,0]
  return summation

def compute_covariance_matrix(x,y,mean_mat,var_mat):
  z = np.asmatrix(np.zeros((x.shape[1],x.shape[1]),dtype=float,order='F'))
  for i in range(x.shape[1]):
    z[i,i] = mean_mat[0,i]*mean_mat[0,i] + var_mat[0,i]
  return z

def main():
  (x,y) = read_params()
  num_classes = 2
  mean_matrix = compute_mean(x,y,num_classes)
  variance_matrix = compute_var(x,y,num_classes)
  mean_0 = mean_matrix[0,:]
  mean_1 = mean_matrix[1,:]
  var_0 = variance_matrix[0,:]
  var_1 = variance_matrix[1,:]
  covariance_matrix = compute_covariance_matrix(x,y,mean_0,var_0)

if __name__ == "__main__":
	main()