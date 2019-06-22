import re
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import bigrams
from nltk.stem import WordNetLemmatizer

# Necessary functionas copied from utils.py
def _stem(doc, p_stemmer, en_stop, return_tokens,feature_name):
    if feature_name=="bigram":
    	tokens = word_tokenize(doc.lower())
    	stemmed_tokens = bigrams(tokens)
    elif feature_name=="lemma":
    	lemmatizer = WordNetLemmatizer()
    	stemmed_tokens = lemmatizer.lemmatize(doc.lower())
    else:
    	tokens = word_tokenize(doc.lower())
    	stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    	stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)

def getStemmedDocuments(docs,feature_name,return_tokens=True):
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_stem(item, p_stemmer, en_stop, return_tokens,feature_name))
        return output_docs
    else:
        return _stem(docs, p_stemmer, en_stop, return_tokens,feature_name)

# reading and cleansing file specified by the given path
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
		train_X[counter] = review_text
		train_Y[counter] = int(line_contents["stars"])
		counter+=1
	return (train_X,train_Y)

# drawing the confusion matrix
def draw_confusion(confatrix):
	plt.imshow(confatrix)
	plt.title("Confusion Matrix")
	plt.colorbar()
	plt.set_cmap("Greens")
	plt.ylabel("True labels")
	plt.xlabel("Predicted label")
	plt.show()

# generating the dictionary
# Two ways either include the raw words or use the words after reducing them to stem
# Class occurences store the number of times each class has occured
# Class vocabulary store the number of words which occured in any document corresponding to that class
def generate_dictionary(train_X,train_Y,feature_name):
	dictionary = {}
	num_words = len(train_X)
	class_occurences = [0,0,0,0,0]
	class_vocabulary = [0,0,0,0,0]
	if feature_name=="split":
		for i in range(len(train_X)):
			num_stars = train_Y[i]
			splitted_string = train_X[i].split()
			class_vocabulary[num_stars-1]+=len(splitted_string)
			class_occurences[num_stars-1]+=1
			for word in splitted_string:
				if word in dictionary:
					dictionary[word][num_stars-1]+=1
				else:
					dictionary[word] = [1,1,1,1,1]
					dictionary[word][num_stars-1]+=1
	else:
		for i in range(len(train_X)):
			num_stars = train_Y[i]
			splitted_string = getStemmedDocuments(train_X[i],feature_name,True)
			class_vocabulary[num_stars-1]+=len(splitted_string)
			class_occurences[num_stars-1]+=1
			for word in splitted_string:
				if word in dictionary:
					dictionary[word][num_stars-1]+=1
				else:
					dictionary[word] = [1,1,1,1,1]
					dictionary[word][num_stars-1]+=1

	return (dictionary,class_occurences,class_vocabulary)

# prediction function
def predict(dictionary,test_X,test_Y,class_occurences,class_vocabulary,feature_name):
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

	if feature_name=="split":
		for i in range(num_test_points):
			prob = [0.0,0.0,0.0,0.0,0.0]
			for j in range(5):
				prob[j]+=class_probabilities[j]
				splitted_string = test_X[i].split()
				for word in splitted_string:
					if word in dictionary:
						prob[j]+=dictionary[word][j]
					else:
						prob[j]-=math.log((float(num_distinct_words+class_vocabulary[j])))
			prediction[i] = 1+np.argmax(prob)
	else:
		for i in range(num_test_points):
			prob = [0.0,0.0,0.0,0.0,0.0]
			for j in range(5):
				prob[j]+=class_probabilities[j]
				splitted_string = getStemmedDocuments(test_X[i],feature_name,True)
				for word in splitted_string:
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
  	
  	if part=='a':
		# reading training and test data from .json file
  		(train_X,train_Y) = read_file(train_data_path)
  		(test_X,test_Y) = read_file(test_data_path)
  		
	  	# creating the dictionary i.e. for keeping count of words
	  	(dictionary,class_occurences,class_vocabulary) = generate_dictionary(train_X,train_Y,"split")
	  	
	  	# prediction time on test_X
	  	prediction = predict(dictionary,test_X,test_Y,class_occurences,class_vocabulary,"split")
	  	
	  	test_Y_array = [0]*len(test_Y)
	  	for i in range(len(test_Y)):
	  		test_Y_array[i] = test_Y[i]

	  	confatrix = confusion_matrix(test_Y_array,prediction)
	  	f1_matrix = f1_score(test_Y_array,prediction,average=None)
	  	print("F1 Score")
	  	print(f1_matrix)
	  	print("Confusion Matrix")
	  	print(confatrix)
	  	macro_f1 = f1_score(test_Y_array,prediction,average='macro')
	  	print("Macro F1 Score")
	  	print(macro_f1)
	  	# draw_confusion(confatrix)

	elif part=='b':
		# means random and majority prediction
		# reading training and test data from .json file
  		(train_X,train_Y) = read_file(train_data_path)
  		(test_X,test_Y) = read_file(test_data_path)
  		
  		# creating the dictionary i.e. for keeping count of words
	  	(dictionary,class_occurences,class_vocabulary) = generate_dictionary(train_X,train_Y,"split")
	  	
	  	test_Y_array = [0]*len(test_Y)
	  	for i in range(len(test_Y)):
	  		test_Y_array[i] = test_Y[i]

	  	max_occuring_class = 1+np.argmax(class_occurences)
  		majority_prediction = [max_occuring_class] * len(test_X)
  		confatrix = confusion_matrix(test_Y_array,majority_prediction)
  		print("Confusion Matrix for majority prediction")
	  	print(confatrix)
	  	# draw_confusion(confatrix)

	  	random_prediction = np.random.random_integers(1,5,(len(test_X),1))
	  	confatrix = confusion_matrix(test_Y_array,random_prediction)
	  	print("Confusion Matrix for random prediction")
	  	print(confatrix)
	  	# draw_confusion(confatrix)

	elif part=='d':
		# reading training and test data from .json file
  		(train_X,train_Y) = read_file(train_data_path)
  		(test_X,test_Y) = read_file(test_data_path)
  		
  		# creating the dictionary i.e. for keeping count of words
	  	(dictionary,class_occurences,class_vocabulary) = generate_dictionary(train_X,train_Y,"stemming")
	  	
	  	# prediction time on test_X
	  	prediction = predict(dictionary,test_X,test_Y,class_occurences,class_vocabulary,"stemming")
	  	
	  	test_Y_array = [0]*len(test_Y)
	  	for i in range(len(test_Y)):
	  		test_Y_array[i] = test_Y[i]

	  	confatrix = confusion_matrix(test_Y_array,prediction)
	  	f1_matrix = f1_score(test_Y_array,prediction,average=None)
	  	macro_f1 = f1_score(test_Y_array,prediction,average='macro')
	  	print("Confusion Matrix")
	  	print(confatrix)
	  	print("F1 Score")
	  	print(f1_matrix)
	  	print("Macro F1 Score")
	  	print(macro_f1)
	  	# draw_confusion(confatrix)

	elif part=='e':
		# feature_name = "stemming"
		feature_name = "split"
		
		# reading training and test data from .json file
  		(train_X,train_Y) = read_file(train_data_path)
  		(test_X,test_Y) = read_file(test_data_path)
  		
  		# creating the dictionary i.e. for keeping count of words
	  	(dictionary,class_occurences,class_vocabulary) = generate_dictionary(train_X,train_Y,feature_name)
	  	
	  	# prediction time on test_X
	  	prediction = predict(dictionary,test_X,test_Y,class_occurences,class_vocabulary,feature_name)
	  	
	  	test_Y_array = [0]*len(test_Y)
	  	for i in range(len(test_Y)):
	  		test_Y_array[i] = test_Y[i]

	  	confatrix = confusion_matrix(test_Y_array,prediction)
	  	print("Confusion Matrix")
	  	print(confatrix)
	  	f1_matrix = f1_score(test_Y_array,prediction,average=None)
	  	print("F1 Score")
	  	print(f1_matrix)
	  	macro_f1 = f1_score(test_Y_array,prediction,average='macro')
	  	print("Macro F1 Score")
	  	print(macro_f1)
	  	# draw_confusion(confatrix)

	else:
		print("No such part")

if __name__ == "__main__":
	main()


