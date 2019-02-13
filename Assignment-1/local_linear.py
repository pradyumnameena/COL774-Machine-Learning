import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

def read_params():
  x_data = pd.read_csv("weightedX.csv",header=None)
  y_data = pd.read_csv("weightedY.csv",header=None)
  x = np.asmatrix(np.array(x_data))
  y = np.asmatrix(np.array(y_data))
  x = normalize(x)
  return (x,y,np.array(x_data),np.array(y_data))

def normalize(mat):
  z = 1+np.zeros((mat.shape[0],1),dtype=float)
  z = np.hstack((z,mat))
  return z

def exp_function(x1,x):
  tow = 0.8
  diff = x1-x
  z = np.dot(diff,diff.transpose())[0,0]
  rv = np.exp((-1*z)/(2*tow*tow))
  return rv

def gradient(x,y,theta,point_x,point_y,weight):
  arr = np.multiply(weight,np.dot(x,theta)-y)
  grad = (np.dot(x.transpose(),arr))/len(x)
  return grad

def local_weighted(point_x,point_y,x,y):
  alpha = 0.03
  old_val = 1000
  new_val = 100
  epsilon = 0.0001
  theta = np.asmatrix(np.zeros((x.shape[1],1),dtype=float,order='F'))
  
  weight = np.asmatrix(np.zeros((x.shape[0],1),dtype=float,order='F'))
  for i in range(len(x)):
      weight[i,0] = exp_function(x[i,:],point_x)
  
  while True:
    grad = gradient(x,y,theta,point_x,point_y,weight)
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

def curve_plot(x_data,y_data,value_mat):
  plt.scatter(x_data,y_data,c='r',label='original')
  new_x,new_y = zip(*sorted(zip(x_data,np.array(value_mat))))
  plt.plot(new_x,new_y,'-',linewidth=2,c='g',label='predicted')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title("local_lin_reg bandwidth = 0.8")
  plt.legend()
  plt.savefig('local_linear.png',dpi=200)

def main():
  (x,y,x_data,y_data) = read_params()
  value_mat = algo(x,y)
  curve_plot(x_data,y_data,value_mat)

if __name__ == "__main__":
	main()