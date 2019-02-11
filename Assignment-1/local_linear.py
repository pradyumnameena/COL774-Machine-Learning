import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from decimal import Decimal

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
    for j in range (mat.shape[1]):
      mat[i,j] = Decimal(mat[i,j]) - Decimal(mean[0,j])
      mat[i,j] = Decimal(mat[i,j])/Decimal(var[0,j])
  z = np.zeros((mat.shape[0],1),dtype=float)
  z+=1
  z = np.hstack((z,mat))
  return z

def exp_function(x1,x):
  tow = 0.8
  diff = x1-x
  diff_sqr = np.multiply(diff,diff)
  z = np.sum(diff_sqr,axis=1)[0,0]
  rv = np.exp((-1*z)/(2*tow*tow))
  return rv

def gradient(x,y,theta,point_x,point_y):
  diff = np.dot(x,theta) - y
  grad2 = np.asmatrix(np.zeros((x.shape[0],x.shape[1]),dtype=float))
  for i in range(len(diff)):
    dist = exp_function(point_x,x[i,:])
    for j in range (x.shape[1]):
      grad2[i,j] = x[i,j]*diff[i,0]*dist
  grad = np.sum(grad2,axis=0).transpose()
  return grad

def local_weighted(point_x,point_y,x,y):
  theta = np.asmatrix(np.zeros((x.shape[1],1),dtype=float,order='F'))
  num_iter = 300
  alpha = 0.03
  alpha/=len(x)
  for counter in range(num_iter):
    grad = gradient(x,y,theta,point_x,point_y)
    theta-=alpha*grad
  return np.dot(point_x,theta)[0,0]

def algo(x,y):
  output_values = np.asmatrix(np.zeros((x.shape[0]/100,1),dtype=float,order='F'))
  for i in range(len(x)/100):
    output_values[i,0] = local_weighted(x[i,:],y[i,0],x,y)
  return output_values

def main():
  (x,y) = read_params()
  value_mat = algo(x,y)
  print(value_mat)
  print("*******************")
  print(y)

if __name__ == "__main__":
	main()