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

def shouldstop(x,y,theta,num_iter):
  ans = True
  if(num_iter>0):
    ans = False
  return ans

def algo(x,y,theta,alpha):
  num_iter = 1000
  alpha/=len(x)
  stopping_cond = False
  while stopping_cond==False:
    grad = gradient(x,y,theta)
    theta-=alpha*grad
    num_iter-=1
    stopping_cond = shouldstop(x,y,theta,num_iter)
  return

def main():
  (x,y) = read_params()
  theta = np.asmatrix(np.zeros((x.shape[1],1),dtype=float,order='F'))
  print(theta.shape)
  alpha = 0.03
  algo(x,y,theta,alpha)

if __name__ == "__main__":
	main()