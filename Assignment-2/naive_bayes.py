import re
import sys
import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# reading the training data
def read_training_params(train_data_path):
	training_text = pd.read_csv(train_data_path,usecols=['text'],encoding="utf-8")
	training_class = pd.read_csv(train_data_path,usecols=['stars'],dtype=int)
	return (training_text,training_class)

# reading the test data
def read_test_params(test_data_path):
	test_text = pd.read_csv(test_data_path,usecols=['text'],encoding="utf-8")
	test_class = pd.read_csv(test_data_path,usecols=['stars'],dtype=int)
	return (test_text,test_class)

# computing accuracy from confusion matrix
def accuracy(confusion_matrix):
	total_points = np.sum(confusion_matrix)
	req = 0
	for i in range(len(confusion_matrix)):
		req+=confusion_matrix[i,i]
	return (float(req)/float(total_points))

# calculates the occurences of each class
def get_class_count(class_array):
	count_arr = np.array([0,0,0,0,0])
	for i in range(5):
		count_arr[i] = np.sum(np.multiply(class_array,(class_array==(float(i+1)))))/(i+1)
	return count_arr

# get naive split tokenized count
def get_split_count(text_matrix,class_matrix):
	word_count = {}
	num_points = text_matrix.shape[0]
	for i in range(num_points):
		# splitted_string = re.sub('[^a-zA-Z0-9\s]','',text_matrix.iat[i,0]).split()
		splitted_string = text_matrix.iat[i,0].split()
		class_idx = class_matrix.iat[i,0]-1
		for word in splitted_string:
			if word in word_count:
				word_count[word][class_idx]+=1
			else:
				word_count[word] = [0,0,0,0,0]
				word_count[word][class_idx]+=1		
	return word_count

# for naive split test predictions
def testing_time_split(word_count,count_arr,test_text,test_class):
	num_points = test_class.shape[0]
	prediction = np.array(np.zeros((num_points,1),dtype=int))
	total_points = np.sum(count_arr)
	class_vocab_size = np.asmatrix(np.zeros((1,5),dtype=int))
	num_words = 0
	
	for word in word_count:
		for i in range(5):
			class_vocab_size[0,i]+=word_count[word][i]
		num_words+=1

	for i in range(num_points):
		print(i)
		prob_arr = np.log(count_arr/float(total_points))
		# splitted_string = re.sub('[^a-zA-Z0-9\s]','',test_text.iat[i,0]).split()
		splitted_string = test_text.iat[i,0].split()
		for word in splitted_string:
			# if word in word_count:
				for j in range(5):
					prob_arr[j] += np.log(float(word_count[word][j]+1)/float(class_vocab_size[0,j]+num_words))
			# else:
				# for j in range(5):
					# prob_arr[j] -= np.log(float(class_vocab_size[0,j]+num_words))
		# assigning the label with maximum probabilities
		prediction[i] = 1+np.argmax(prob_arr)
	return prediction

# main function
def main():
	
	# Taking parameters from command line
	train_data_path = sys.argv[1]
  	test_data_path = sys.argv[2]
  	part = sys.argv[3]

  	# reading the dataset from the already converted csv file into pandas dataframe
	(training_text,training_class) = read_training_params(train_data_path)
	(test_text,test_class) = read_test_params(test_data_path)

	# preprocessing the text data returns the class occurences
	if part == 'a':
		training_class_counts = get_class_count(np.array(training_class))
		word_class_count = get_split_count(training_text,training_class)
		prediction = testing_time_split(word_class_count,training_class_counts,test_text,test_class)
		true_label_array = np.array(test_class)
	
		confatrix = confusion_matrix(true_label_array,prediction)
		print(confatrix)
		print("Accuracy obtained is " + str(accuracy(confatrix)))
		f1_matrix = f1_score(true_label_array,prediction,average=None)
		print(f1_matrix)
	
	elif part == 'b':
		# means random prediction and majority prediction
		training_class_counts = get_class_count(np.array(training_class))
		num_points = test_class.shape[0]
		random_prediction = np.random.randint(1,6,(num_points,1))
		majority_prediction = (1+np.argmax(training_class_counts))*np.ones((num_points,1),dtype=int)
		true_label_array = np.array(test_class[0:num_points])
		
		random_confatrix = confusion_matrix(true_label_array,random_prediction)
		print("Accuracy obtained for random predictions is " + str(accuracy(random_confatrix)))
		f1_matrix = f1_score(true_label_array,random_prediction,average=None)
		print(f1_matrix)

		majority_confatrix = confusion_matrix(true_label_array,majority_prediction)
		print("Accuracy obtained for majority predictions is " + str(accuracy(majority_confatrix)))
		f1_matrix = f1_score(true_label_array,majority_prediction,average=None)
		print(f1_matrix)
	else:
		print("fnkjf")

if __name__ == "__main__":
 	main()