import numpy as np
import pandas as pd
import math
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# from utils.py which was supplied in the package provided
def _stem(doc, p_stemmer, en_stop, return_tokens):
    tokens = word_tokenize(doc.lower())
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)

def getStemmedDocuments(docs, return_tokens=True):
    """
        Args:
            docs: str/list(str): document or list of documents that need to be processed
            return_tokens: bool: return a re-joined string or tokens
        Returns:
            str/list(str): processed document or list of processed documents
        Example: 
            new_text = "It is important to by very pythonly while you are pythoning with python. \
                All pythoners have pythoned poorly at least once."
            print(getStemmedDocuments(new_text))
        Reference: https://pythonprogramming.net/stemming-nltk-tutorial/
    """
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_stem(item, p_stemmer, en_stop, return_tokens))
        return output_docs
    else:
        return _stem(docs, p_stemmer, en_stop, return_tokens)

# reading the parameters
def read_params():
	# text is read in a matrix so that it is easy to pass it into countvectorizer
	text_matrix = pd.read_csv('ass2_data/train.csv',usecols=['text'],encoding="utf-8")
	class_matrix = pd.read_csv('ass2_data/train.csv',usecols=['stars'],dtype=int)
	return (text_matrix,class_matrix)

# reading the test parameters
def read_test_params():
	test_text = pd.read_csv('ass2_data/test.csv',usecols=['text'],encoding="utf-8")
	test_class = pd.read_csv('ass2_data/test.csv',usecols=['stars'],dtype=int)
	return (test_text,test_class)

# calculates the occurences of each class
def get_class_count(class_matrix):
	# for probability of various classes
	class_matrix= np.asmatrix(class_matrix)
	num1 = np.sum(np.multiply(class_matrix,(class_matrix==1.0)))
	num2 = np.sum(np.multiply(class_matrix,(class_matrix==2.0)))/2
	num3 = np.sum(np.multiply(class_matrix,(class_matrix==3.0)))/3
	num4 = np.sum(np.multiply(class_matrix,(class_matrix==4.0)))/4
	num5 = np.sum(np.multiply(class_matrix,(class_matrix==5.0)))/5
	count_arr = np.array([num1,num2,num3,num4,num5])
	return (count_arr)

# get naive split tokenized count
def get_split_count(text_matrix,class_matrix):
	word_count = np.asmatrix(np.zeros((5,1),dtype=int,order='F'))
	word_counter = 0
	word_index_mapping = {}
	num_points = text_matrix.shape[0]
	
	for i in range(num_points):
		splitted_string = re.sub('[^a-zA-Z0-9\s]','',text_matrix.iat[i,0]).split()
		class_idx = class_matrix.iat[i,0]-1
		for word in splitted_string:
			if word_index_mapping.has_key(word)==False:
				# first occurence of word hence update the dictionary as well
				word_count = np.hstack((word_count,np.zeros((5,1),dtype=int,order='F')))
				word_index_mapping.update({word:word_counter})
				word_count[class_idx,word_counter]+=1
				word_counter+=1
			else:
				# word has already been seen hence update the matrix
				word_count[class_idx,word_index_mapping.get(word)]+=1
	return (word_count,word_index_mapping,word_counter)

# for naive split test predictions
def testing_time_naive_split(word_count,index_mapping,count_arr,test_text,test_class,num_words,num_points):
	prediction = np.array(np.zeros((num_points,1),dtype=int))
	total_points = np.sum(count_arr)
	class_vocab_size = np.sum(word_count,axis=1)
	for i in range(num_points):
		prob_arr = [0.0,0.0,0.0,0.0,0.0]
		splitted_string = re.sub('[^a-zA-Z0-9\s]','',test_text.iat[i,0]).split()
		for word in splitted_string:
			if index_mapping.has_key(word):
				word_idx = index_mapping.get(word)
				for j in range(5):
					prob_arr[j] += (math.log10(float(word_count[j,word_idx] + 1)) - math.log10(float(class_vocab_size[j] + num_words)))
			else:
				for j in range(5):
					prob_arr[j] -= math.log10(float(class_vocab_size[j] + num_words))
					
		# class probabilities
		for j in range(5):
			prob_arr[j] += math.log10(float(count_arr[j])/float(total_points))
		# assigning the label with maximum probabilities
		prediction[i] = 1+np.argmax(prob_arr)
	return prediction

# main function
def main():
	# reading the dataset from the already converted csv file into pandas dataframe
	(text_matrix,class_matrix) = read_params()
	print("successfully read training data")

	# preprocessing the text data returns the class occurences
	count_arr = get_class_count(np.array(class_matrix))
	print("computed class counts")
	
	# returns the word count distribution across classes
	(word_count,index_mapping,word_counter) = get_split_count(text_matrix,class_matrix)
	print("computed distribution of tokens across classes")
	
	# loading the test data
	(test_text,test_class) = read_test_params()
	print("successfully read test data")
	
	num_points = test_class.shape[0]
	# calculating on test data
	prediction = testing_time_naive_split(word_count,index_mapping,count_arr,test_text,test_class,word_counter,num_points)
	print("prediction done")

	# calculating the confusion matrix
	true_label_array = np.array(test_class[0:num_points])
	confatrix = confusion_matrix(true_label_array,prediction)
	print("confusion matrix computed")
	print(confatrix)

	# calculating f1 score across various classes
	f1_matrix = f1_score(true_label_array,prediction,average=None)
	print("f1_score for separate class calculated separately")
	print(f1_matrix)
	return

if __name__ == "__main__":
 	main()