import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

def read_params():
	x_data = pd.read_csv("ass1_data/linearX.csv",header=None)
	y_data = pd.read_csv("ass1_data/linearY.csv",header=None)
	return (x_data,y_data)

def matrix_mult(A,B):
	out = linalg(A*B)
	return out

def compute_gradient(x,y,theta):
	grad = np.zeros(x.shape[1]+1,dtype=float,order='F')
	print(x.shape)
	print(theta.shape)
	prediction = matrix_mult(x,theta)
	return grad

def cost(x,y,theta):

	return 0

def main():
	(x,y) = read_params()
	theta = np.zeros((x.shape[1]+1,1),dtype=float,order='F')
	compute_gradient(x,y,theta)

if __name__ == "__main__":
	main()