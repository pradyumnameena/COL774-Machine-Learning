import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

def read_params(x_path,y_path):
  x_data = pd.read_csv(x_path,header=None)
  y_data = pd.read_csv(y_path,header=None)
  x = np.asmatrix(np.array(x_data))
  y = np.asmatrix(np.array(y_data))
  x = normalize(x)
  return (x,y,np.array(x_data),np.array(y_data))

def normalize(mat):
  # Prof said no need for normalization, just add the initial column
  z = 1+np.zeros((mat.shape[0],1),dtype=float)
  z = np.hstack((z,mat))
  return z

# computes the distance of x1 from x parametrized by tow
def exp_function(x1,x,tow):
  diff = x1-x
  z = np.dot(diff,diff.transpose())[0,0]
  rv = np.exp((-1*z)/(2*tow*tow))
  return rv

def gradient(x,y,theta,point_x,point_y,weight):
  arr = np.multiply(weight,np.dot(x,theta)-y)
  grad = (np.dot(x.transpose(),arr))/len(x)
  return grad

def local_weighted(point_x,point_y,x,y,tow):
  
  # this alpha is for the underlying gradient descent for learning params for a given point
  alpha = 0.03
  old_val = 10
  new_val = 1
  epsilon = 0.0001
  
  # theta is the vector of parameters for predicting on point_x
  theta = np.asmatrix(np.zeros((x.shape[1],1),dtype=float,order='F'))
  
  # assigning weight to the m data-points
  weight = np.asmatrix(np.zeros((x.shape[0],1),dtype=float,order='F'))
  for i in range(len(x)):
      weight[i,0] = exp_function(x[i,:],point_x,tow)
  
  # convergence criteria: change in predicted value<epsilon
  while True:
    grad = gradient(x,y,theta,point_x,point_y,weight)
    theta-=alpha*grad
    new_val = np.dot(point_x,theta)
    if(abs(new_val-old_val)<epsilon):
      break
    old_val = new_val

  # [0,0] because the outout will be a 1x1 matrix and we need scalar output
  return np.dot(point_x,theta)[0,0]

def algo(x,y,tow):
  
  # Since no test set was given. I applied the algo on the given test set only
  # output_values contains the predicted value on the given dataset
  output_values = np.asmatrix(np.zeros((x.shape[0],1),dtype=float,order='F'))
  for i in range(len(x)):
    output_values[i,0] = local_weighted(x[i,:],y[i,0],x,y,tow)
  return output_values

def curve_plot(x_data,y_data,value_mat):
  plt.scatter(x_data,y_data,c='r',label='Original')
  new_x,new_y = zip(*sorted(zip(x_data,np.array(value_mat))))
  plt.plot(new_x,new_y,'-',linewidth=2,c='g',label='Predicted')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title("local_lin_reg")
  plt.legend()
  plt.savefig('local_linear.png',dpi=200)

def main():

  # Taking parameters from command line
  x_path = sys.argv[1]
  y_path = sys.argv[2]
  tow = float(sys.argv[3])
  
  # reading the datasets
  (x,y,x_data,y_data) = read_params(x_path,y_path)

  print("Since no dataset was supplied for testing I am testing on the given dataset only.")
  print("If you want to see the result uncomment line 94 and line 97")

  # output values on the given dataset when predicted using locally weighted linear regression
  # value_mat = algo(x,y,tow)
  
  # plotting
  # curve_plot(x_data,y_data,value_mat)

if __name__ == "__main__":
	main()