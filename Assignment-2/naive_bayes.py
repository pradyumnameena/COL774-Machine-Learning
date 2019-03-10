import re
import sys
import json
import time
import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def read_file(file_path):
	train_X = {}
	train_Y = {}
	counter = 0

	for line in open(file_path, mode="r"):
		line_contents = json.loads(line)
		review_text = line_contents["text"].strip().lower()
		review_text = re.sub(r'[^\w\s]','',review_text)
		review_text = re.sub(r'[^\w\s]','',review_text)
		review_text = re.sub('\r?\n',' ',review_text)
		splitted_string = review_text.split()
		train_X[counter] = splitted_string
		train_Y[counter] = int(line_contents["stars"])
		counter+=1
	return (train_X,train_Y)

def generate_dictionary(train_X,train_Y):
	dictionary = {}
	num_words = len(train_X)
	class_occurences = [0,0,0,0,0]
	class_vocabulary = [0,0,0,0,0]

	for i in range(len(train_X)):
		num_stars = train_Y[i]
		class_vocabulary[num_stars-1]+=len(train_X[i])
		class_occurences[num_stars-1]+=1
		for word in train_X[i]:
			if word in dictionary:
				dictionary[word][num_stars-1]+=1
			else:
				dictionary[word] = [1,1,1,1,1]
				dictionary[word][num_stars-1]+=1

	return (dictionary,class_occurences,class_vocabulary)

def predict(dictionary,test_X,test_Y,class_occurences,class_vocabulary):
	num_test_points = len(test_X)
	num_train_points = sum(class_occurences)
	num_distinct_words = len(dictionary)
	prediction = [0] * num_test_points
	class_probabilities = [0.0]*5
	for j in range(5):
		class_probabilities[j]=math.log((float(class_occurences[j]))/(float(num_train_points)))

	for word in dictionary:
		for j in range(5):
			dictionary[word][j] = math.log((float(dictionary[word][j]))/(float(num_distinct_words+class_vocabulary[j])))

	for i in range(num_test_points):
		prob = [0.0,0.0,0.0,0.0,0.0]
		for j in range(5):
			prob[j]+=class_probabilities[j]
			for word in test_X[i]:
				if word in dictionary:
					prob[j]+=dictionary[word][j]
				else:
					prob[j]-=math.log((float(num_distinct_words+class_vocabulary[j])))
		prediction[i] = 1+np.argmax(prob)
	return prediction

# main function
def main():
	# Taking parameters from command line
	train_data_path = sys.argv[1]
  	test_data_path = sys.argv[2]
  	part = sys.argv[3]
  	
  	# reading training and test data from .json file
  	time1 = time.clock()
  	(train_X,train_Y) = read_file(train_data_path)
  	time2 = time.clock()
  	print(str(time2-time1) + " reading training json file")
  	
  	time1 = time.clock()
  	(test_X,test_Y) = read_file(test_data_path)
  	time2 = time.clock()
  	print(str(time2-time1) + " reading testing json file")

  	# creating the dictionary i.e. for keeping count of words
  	time1 = time.clock()
  	(dictionary,class_occurences,class_vocabulary) = generate_dictionary(train_X,train_Y)
  	time2 = time.clock()
  	print(str(time2-time1) + " generating vocabulary")

  	# prediction time on test_X
  	time1 = time.clock()
  	prediction = predict(dictionary,test_X,test_Y,class_occurences,class_vocabulary)
  	time2 = time.clock()
  	print(str(time2-time1) + " for prediction")
  	
  	test_Y_array = [0]*len(test_Y)
  	for i in range(len(test_Y)):
  		test_Y_array[i] = test_Y[i]

  	confatrix = confusion_matrix(test_Y_array,prediction)
  	print(confatrix)
  	# print((float(prediction))/(float(len(test_Y))) * 100)

if __name__ == "__main__":
 	main()