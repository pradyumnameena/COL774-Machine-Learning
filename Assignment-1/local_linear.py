import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

def exp_function(x1,x,tow):
  diff = x1-x
  diff_sqr = np.multiply(diff,diff)
  z = np.sum(diff_sqr,axis=0)
  rv = math.exp(-z/(2*tow*tow))
  return rv

def read_params():
  x_data = pd.read_csv("weightedX.csv",header=None)
  y_data = pd.read_csv("weightedY.csv",header=None)
  x = np.asmatrix(np.array(x_data))
  y = np.asmatrix(np.array(y_data))
  x = normalize(x)
  return (x,y)

def normalize(mat):
  mean = np.mean(mat,axis=0)
  var = np.var(mat,axis=0)
  for i in range (len(mat)):
    mat[i,:] = mat[i,:] - mean
    mat[i,:] = mat[i,:]/var
  z = np.zeros((mat.shape[0],1),dtype=float)
  z+=1
  z = np.hstack((z,mat))
  return z

def gradient(x,y,theta,point_x,point_y):
  diff = np.dot(x,theta) - y
  grad2 = np.asmatrix(np.zeros((x.shape[0],x.shape[1]),dtype=float))
  for i in range(len(diff)):
    dist = exp_function(point_x,x[i,:],tow)
    grad2[i,:] = np.multiply(x[i,:],diff[i]*dist)
  grad = np.sum(grad2,axis=0).transpose()
  return grad

def local_weighted(point_x,point_y,x,y):
  theta = np.asmatrix(np.zeros((x.shape[1],1),dtype=float,order='F'))
  num_iter = 100
  alpha = 0.003
  for counter in range(num_iter):
    grad = gradient(x,y,theta,point_x,point_y)
    theta-=(alpha/len(x))*grad
  return np.dot(point_x,theta)

def algo(x,y):
  # add code for looping through all the points
  # for local_weighted function use the row vector from the X matrix
  output_values = np.asmatrix(np.zeros((x.shape[0],1),dtype=float,order='F'))
  for i in range(len(x)):
    output_values[i] = local_weighted(x[i,:],y[i],x,y)
  return

def main():
  (x,y) = read_params()
  theta = np.asmatrix(np.zeros((x.shape[1],1),dtype=float,order='F'))
  alpha = 0.03
  algo(x,y,theta,alpha)

if __name__ == "__main__":
	main()