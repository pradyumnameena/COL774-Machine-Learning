import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from decimal import Decimal

def read_params(x_path,y_path):
  x_data = pd.read_csv(x_data,header=None)
  y_data = pd.read_csv(y_path,header=None)
  x = normalize(np.asmatrix(np.array(x_data)))
  y = np.asmatrix(np.array(y_data))
  return (x,y,np.array(x_data),np.array(y_data))

def normalize(mat):

  # x = (x-mean)/variance
  mean = np.mean(mat,axis=0)
  var = np.var(mat,axis=0)
  mat-=mean
  for i in range (len(mat)):
  	mat[i,:] = np.divide(mat[i,:],var)

  # including the initial column of x0 feature
  z = 1 + np.zeros((mat.shape[0],1),dtype=float)
  z = np.hstack((z,mat))
  return z

def compute_cost(x,y,theta):
  diff = y-np.dot(x,theta)
  rv = np.dot(diff.transpose(),diff)
  return rv[0,0]/(2*len(x))

def gradient(x,y,theta):
  grad = np.dot(np.dot(x.transpose(),x),theta) - np.dot(x.transpose(),y)
  return grad/len(x)

def contour_plot(x_data,y_data):
  num_points_x1 = 100
  num_points_x2 = 100
  x1 = np.linspace(-5,5,num_points_x1)
  x2 = np.linspace(-5,5,num_points_x2)
  X1,X2 = np.meshgrid(x1,x2)
  Y = np.asmatrix(np.zeros((100,100),dtype=float))
  for i in range(num_points_x1):
    for j in range(num_points_x2):
      Y[i,j] = compute_cost(x_data,y_data,[[x1[i]],[x2[j]]])
  cp = plt.contour(X1,X2,np.array(Y))
  # plt.show()

def algo(x,y,theta,alpha):
  num_iter = 0
  epsilon = 0.000000001
  old_cost = 1000
  new_cost = 0
  
  # convergence criteria: Change in cost<epsilon
  while True:
    grad = gradient(x,y,theta)
    theta-=alpha*grad
    new_cost = compute_cost(x,y,theta)
    if(old_cost-new_cost<epsilon):
      break
    old_cost = new_cost
    num_iter+=1
  # On console the number of iterations taken and the cost
  print("Number of iterations = " + str(num_iter))
  print("Final cost = " + str(new_cost))

def curve_plot(x,theta,x_data,y_data):
  y_out = np.array(np.dot(x,theta))
  plt.figure(0)
  plt.title('Linear regression')
  plt.scatter(x_data,y_data,c='r',label='Given Data')
  plt.xlabel(' x ')
  plt.ylabel(' y ')
  plt.figure(0)
  new_x,new_y = zip(*sorted(zip(x_data,y_out)))
  plt.plot(new_x,new_y,'-',linewidth=2,c='g',label='Hypothesis')
  plt.legend()
  plt.savefig('linear_reg.png',dpi=200)

def main():
  
  # Taking parameters from command line
  x_path = sys.argv[1]
  y_path = sys.argv[2]
  alpha = float(sys.argv[3])
  time_gap = float(sys.argv[4])
  
  # Reading the dataset
  (x,y,x_data,y_data) = read_params(x_path,y_path)
  
  # Computing the theta vector
  theta = np.asmatrix(np.zeros((x.shape[1],1),dtype=float,order='F'))
  algo(x,y,theta,alpha)
  
  # Plotting
  curve_plot(x,theta,x_data,y_data)
  # contour_plot(x,y)

if __name__ == "__main__":
	main()