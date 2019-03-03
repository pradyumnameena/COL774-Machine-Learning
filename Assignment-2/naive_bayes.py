import numpy as np
import pandas as pd
import csv
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

# calculates the occurences of each class
def pre_processing(class_matrix):
	# for probability of various classes
	num1 = 0
	num2 = 0
	num3 = 0
	num4 = 0
	num5 = 0

	for i in range(class_matrix.shape[0]):
		if class_matrix.iat[i,0] == 1.0 :
			num1+=1
		elif class_matrix.iat[i,0] == 2.0:
			num2+=1
		elif class_matrix.iat[i,0] == 3.0:
			num3+=1
		elif class_matrix.iat[i,0] == 4.0:
			num4+=1
		else:
			num5+=1
	count_arr = np.array([num1,num2,num3,num4,num5])
	# count_arr/=class_matrix.shape[0]
	# using the above instruction leads to the probabilities being exact 0
	return (count_arr)

# get word counts for the tokens obtained after stemming
def get_word_counts(text_matrix,class_matrix):
	word_count = np.asmatrix(np.zeros((5,1),dtype=int,order='F'))
	word_counter = 0
	word_index_mapping = {}
	num_points = text_matrix.shape[0]/100
	for i in range(num_points):
		stemmed_string = getStemmedDocuments(text_matrix.iat[i,0])
		class_idx = class_matrix.iat[i,0]-1
		for word in stemmed_string:
			if word_index_mapping.has_key(word)==False:
				# first occurence of word hence update the dictionary as well
				word_count = np.hstack((word_count,np.zeros((5,1),dtype=int,order='F')))
				word_index_mapping.update({word:word_counter})
				word_count[class_idx,word_counter]+=1
				word_counter+=1
			else:
				# word has already been seen hence update the matrix
				word_count[class_idx,word_index_mapping.get(word)]+=1
	return (word_count,word_index_mapping)

# reading the test parameters
def read_test_params():
	test_text = pd.read_csv('ass2_data/test.csv',usecols=['text'],encoding="utf-8")
	test_class = pd.read_csv('ass2_data/test.csv',usecols=['stars'])
	return (test_text,test_class)

# for computing the confusion matrix
def testing_time(word_count,index_mapping,count_arr,test_text,test_class):
	# getting the total number of words in each class
	total_words_class = np.sum(word_count,axis=1)
	# num_points = test_class.shape[0]
	num_points = 15000
	prediction = np.array(1+np.zeros((num_points,1),dtype=int))
	default_prob = [1.0,1.0,1.0,1.0,1.0]

	for i in range(num_points):
		prob_arr = [1.0,1.0,1.0,1.0,1.0]
		stemmed_string = getStemmedDocuments(test_text.iat[i,0],True)
		# computing for each token separately
		for word in stemmed_string:
			# computing for each class
			for j in range(4):
				if index_mapping.has_key(word) and word_count[j,index_mapping.get(word)]!=0:
					prob_arr[j]*=0.5
					# use the computed values
				else:
					# use default probabilities
					prob_arr[j]*=default_prob[j]
		# assigning the label with maximum probabilities
		prediction[i] = 1+np.argmax(prob_arr)
	return prediction

# main function
def main():
	print("Only 1/100 of dataset is being used on 15000 of datapoints")
	# reading the dataset from the already converted csv file into pandas dataframe
	(text_matrix,class_matrix) = read_params()
	print("successfully read training data")

	# preprocessing the text data returns the class occurences
	count_arr = pre_processing(class_matrix)
	print("computed class counts")
	
	# returns the word count distribution across classes
	(word_count,index_mapping) = get_word_counts(text_matrix,class_matrix)
	print("computed distribution of tokens across classes")
	
	# loading the test data
	(test_text,test_class) = read_test_params()
	print("successfully read test data")

	# calculating on test data
	prediction = testing_time(word_count,index_mapping,count_arr,test_text,test_class)
	print("prediction done")

	true_label_array = np.array(test_class)[0:15000,:]
	# calculating the confusion matrix
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