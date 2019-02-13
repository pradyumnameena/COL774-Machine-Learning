import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from decimal import Decimal

def read_params():
  x_data = pd.read_csv("linearX.csv",header=None)
  y_data = pd.read_csv("linearY.csv",header=None)
  x = np.asmatrix(np.array(x_data))
  y = np.asmatrix(np.array(y_data))
  x = normalize(x)
  x1 = np.array(x_data)
  y1 = np.array(y_data)
  return (x,y,x1,y1)

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

def compute_cost(x,y,theta):
  prod = np.dot(x,theta)
  diff = y-prod
  diff = np.multiply(diff,diff)
  rv = np.sum(diff,axis=0)
  rv/=(2*len(x))
  return rv[0,0]

def gradient(x,y,theta):
  diff = np.dot(x,theta) - y
  grad2 = np.asmatrix(np.zeros((x.shape[0],x.shape[1]),dtype=float))
  for i in range (len(diff)):
   for j in range (x.shape[1]):
	   grad2[i,j] = x[i,j]*diff[i,0]	
  grad = np.sum(grad2,axis=0).transpose()
  return grad

def algo(x,y,theta,alpha):
  num_iter = 0
  alpha/=len(x)
  epsilon = 0.000001
  old_cost = 1000
  new_cost = 0
  while True:
    grad = gradient(x,y,theta)
    theta-=alpha*grad
    new_cost = compute_cost(x,y,theta)
    if(old_cost-new_cost<epsilon):
      break
    old_cost = new_cost
    num_iter+=1
  print("Number of iterations = " + str(num_iter))
  print("Final cost = " + str(new_cost))
  return

def main():
  (x,y,x_data,y_data) = read_params()
  theta = np.asmatrix(np.zeros((x.shape[1],1),dtype=float,order='F'))
  alpha = 0.03
  algo(x,y,theta,alpha)
  print(theta)
  y_out = np.array(np.dot(x,theta))
  # print(x_data[81:100,:])
  # print("Cost on test-set= " + str(compute_cost(x[81:100,:],y[81:100,:],theta)))
  plt.plot(x_data,y_data,'ro')
  plt.plot(x_data,y_out)
  plt.show()

if __name__ == "__main__":
	main()