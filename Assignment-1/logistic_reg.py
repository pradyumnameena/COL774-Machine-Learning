import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from decimal import Decimal

def read_params():
  x_data = pd.read_csv("logisticX.csv",header=None)
  y_data = pd.read_csv("logisticY.csv",header=None)
  x = np.asmatrix(np.array(x_data))
  y = np.asmatrix(np.array(y_data))
  x = normalize(x)
  return (x,y,np.array(x_data),np.array(y_data))

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

def algo(x,y,theta):
  prod = 1/(1+np.exp(-1*np.dot(x,theta)))
  prod1 = np.multiply(y,np.log(prod))
  prod2 = np.multiply(1-y,np.log(1 - prod))
  pred = prod1+prod2
  prod3 = np.multiply(prod,1-prod)
  prod = y - prod
  
  first_der = np.asmatrix(np.zeros((x.shape[0],x.shape[1]),dtype=float,order='F'))
  for i in range(len(x)):
    for j in range(x.shape[1]):
      first_der[i,j] = x[i,j] * prod[i,0]
  first_der = np.sum(first_der,axis=0).transpose()
  
  second_der = np.asmatrix(np.zeros((x.shape[1],x.shape[1]),dtype=float,order='F'))
  for i in range(x.shape[1]):
    for j in range(x.shape[1]):
      for k in range(len(x)):
        second_der[i,j]+=x[k,i]*x[k,j]*prod3[k,0]
  second_der = np.linalg.pinv(-1*second_der)
  
  theta-=np.dot(second_der,first_der)
  return

def main():
  (x,y,x_data,y_data) = read_params()
  theta = np.asmatrix(np.zeros((x.shape[1],1),dtype=float,order='F'))
  algo(x,y,theta)
  print(theta)
  val = np.dot(x,theta)
  val = 1/(1+np.exp(-1*val))
  class_1 = []
  class_2 = []
  for i in range(len(x)):
    if y[i]==1:
      class_1.append([x[i,1],x[i,2]])
    else:
      class_2.append([x[i,1],x[i,2]])
  plt.scatter(np.array(class_1)[:,0],np.array(class_1)[:,1],c='g',label='positive class (y==1)')
  plt.scatter(np.array(class_2)[:,0],np.array(class_2)[:,1],c='r',label='negative class (y==0)')
  x_1 = np.arange(-2,3,0.1)
  y_1 = -(theta[0,0] + theta[1,0]*x_1)/theta[2,0]
  plt.plot(x_1,y_1)
  plt.legend()
  plt.show()
  val = np.hstack((val,y))
  print(val)

if __name__ == "__main__":
	main()