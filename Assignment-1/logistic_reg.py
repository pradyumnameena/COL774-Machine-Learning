import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

# Reading parameters from csv files
def read_params(x_path,y_path):
  x_data = pd.read_csv(x_path,header=None)
  y_data = pd.read_csv(y_path,header=None)
  x = np.asmatrix(np.array(x_data))
  y = np.asmatrix(np.array(y_data))
  x = normalize(x)
  return (x,y,np.array(x_data),np.array(y_data))

# Normalization function
def normalize(mat):
  mean = np.mean(mat,axis=0)
  var = np.var(mat,axis=0)
  mat-=mean
  mat = np.divide(mat,var)
  z = 1 + np.zeros((mat.shape[0],1),dtype=float)
  z = np.hstack((z,mat))
  return z

# Main algorithm for newton's method
def algo(x,y,theta):
  # Current probability of y==1
  prediction = 1/(1+np.exp(-1*np.dot(x,theta)))
  
  # h_1_h = h(x)*(1-h(x))
  h_1_h = np.multiply(prediction,1-prediction)
  diff = y - prediction

  # computing the first derivative
  first_der = np.dot(x.transpose(),diff)
  
  # computing the second derivative
  second_der = np.dot(np.transpose(x),np.multiply(x,h_1_h))
  second_der = np.linalg.pinv(-1*second_der)
  
  # Updating theta vector using newton's method rule
  theta-=np.dot(second_der,first_der)

# Plotting the graph
def curve_plot(x,y,theta):
  # identifying data-points belonging to different classes
  class_1 = []
  class_2 = []
  for i in range(len(x)):
    if y[i]==1:
      class_1.append([x[i,1],x[i,2]])
    else:
      class_2.append([x[i,1],x[i,2]])
  
  # Plotting the given data
  plt.scatter(np.array(class_1)[:,0],np.array(class_1)[:,1],c='g',label='positive class (y==1)')
  plt.scatter(np.array(class_2)[:,0],np.array(class_2)[:,1],c='r',label='negative class (y==0)')
  
  # Plotting the learned curve on the given dataset
  # Note that these two lines are subject to dataset and hence predictions might be different
  x_1 = np.arange(-3,3,0.01)
  y_1 = -(theta[0,0] + theta[1,0]*x_1)/theta[2,0]
  plt.plot(x_1,y_1)
  
  # Customizing the plot
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.legend()
  plt.title('Logistic Regression')
  plt.savefig('logistic_reg.png',dpi=200)
  plt.show()

# MAIN FUNCTION
def main():
  # Taking parameters from command line
  x_path = sys.argv[1]
  y_path = sys.argv[2]

  # Reading the dataset
  (x,y,x_data,y_data) = read_params(x_path,y_path)
  
  # Learning the theta vector
  theta = np.asmatrix(np.zeros((x.shape[1],1),dtype=float,order='F'))
  algo(x,y,theta)

  # Contains the probability of this data-point belonging to positive class i.e. y==1
  val = 1/(1+np.exp(-1*np.dot(x,theta)))
  
  # Plotting
  curve_plot(x,y,theta)

if __name__ == "__main__":
	main()