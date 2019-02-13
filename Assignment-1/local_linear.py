import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from decimal import Decimal
import itertools

def read_params():
  x_data = pd.read_csv("weightedX.csv",header=None)
  y_data = pd.read_csv("weightedY.csv",header=None)
  x = np.asmatrix(np.array(x_data))
  y = np.asmatrix(np.array(y_data))
  x = normalize(x)
  return (x,y,np.array(x_data),np.array(y_data))

def normalize(mat):
  z = np.zeros((mat.shape[0],1),dtype=float)
  z+=1
  z = np.hstack((z,mat))
  return z

def exp_function(x1,x):
  tow = 10
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
  alpha = 0.03
  alpha/=len(x)
  old_val = 1000
  new_val = 100
  epsilon = 0.001
  while True:
    grad = gradient(x,y,theta,point_x,point_y)
    theta-=alpha*grad
    new_val = np.dot(point_x,theta)
    if(abs(new_val-old_val)<epsilon):
      break
    old_val = new_val
  return np.dot(point_x,theta)[0,0]

def algo(x,y):
  output_values = np.asmatrix(np.zeros((x.shape[0],1),dtype=float,order='F'))
  for i in range(len(x)):
    output_values[i,0] = local_weighted(x[i,:],y[i,0],x,y)
  return output_values

def main():
  (x,y,x_data,y_data) = read_params()
  value_mat = algo(x,y)
  
  plt.scatter(x_data,y_data,c='r',label='original')
  new_x,new_y = zip(*sorted(zip(x_data,np.array(value_mat))))
  plt.plot(new_x,new_y,'-',linewidth=2,c='g',label='predicted')
  plt.title("bandwidth = 10")
  plt.legend()
  plt.show()

if __name__ == "__main__":
	main()