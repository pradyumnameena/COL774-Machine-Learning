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
  mat-=mean
  for i in range (len(mat)):
    mat[i,:] = np.divide(mat[i,:],var)
  z = 1 + np.zeros((mat.shape[0],1),dtype=float)
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

def compute_covariance_matrix(class_data,mean_mat,var_mat):
  z = np.dot((class_data-mean_mat).transpose(),(class_data-mean_mat))
  return z/len(class_data)

def curve_plot(class_0,class_1):
  plt.scatter(np.array(class_1)[:,1],np.array(class_1)[:,2],c='g',label='Alaska (y==1)')
  plt.scatter(np.array(class_0)[:,1],np.array(class_0)[:,2],c='r',label='Canada (y==0)')
  plt.legend()
  plt.xlabel('Growth ring diameters in fresh water')
  plt.ylabel('Growth ring diameters in marine water')
  plt.title('Gaussian Discriminant Analysis')
  plt.savefig('gda.png',dpi=200)

def main():
  (x,y) = read_params()
  class_0 = []
  class_1 = []
  for i in range(len(x)):
    if y[i]==0:
      class_0.append([x[i,0],x[i,1],x[i,2]])
    else:
      class_1.append([x[i,0],x[i,1],x[i,2]])

  mean_0 = np.mean(class_0,axis=0)
  mean_1 = np.mean(class_1,axis=0)
  var_0 = np.var(class_0,axis=0)
  var_1 = np.var(class_1,axis=0)
  covariance_matrix_0 = compute_covariance_matrix(class_0,mean_0,var_0)
  covariance_matrix_1 = compute_covariance_matrix(class_1,mean_1,var_1)
  covariance_matrix = compute_covariance_matrix(x,np.mean(x,axis=0),np.var(x,axis=0))
  print(covariance_matrix_0)
  print(covariance_matrix_1)
  # curve_plot(class_0,class_1)
  
if __name__ == "__main__":
	main()