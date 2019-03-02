import numpy as np
import pandas as pd
import csv
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

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

def read_params():
	text_matrix = pd.read_csv('ass2_data/train.csv',usecols=['text'])
	class_matrix = pd.read_csv('ass2_data/train.csv',usecols=['stars'])
	return (text_matrix,class_matrix)

def pre_processing(text_matrix,class_matrix):
	# for preprocessing the text present in the data
	# for stemming

	# for means of various classes
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
	return (text_matrix,count_arr)

def get_word_counts(text_matrix):
	# either use the countvector class or implement your own
	# assumes that stemming has already been done
	key_words = 10000
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(text_matrix.to_numpy())
	print(text_matrix)
	print(X)
	word_count = X.toarray()
	# word_count = np.asmatrix(np.zeros((key_words,5),dtype=int,order='F'))
	return word_count

def main():
	
	# reading the dataset from the already converted csv file
	(text_matrix,class_matrix) = read_params()

	# preprocessing the text data i.e. stemming and all
	# also return the class occurences
	(text_matrix,count_arr) = pre_processing(text_matrix,class_matrix)
	
	# returns the word count distribution across classes
	word_count = get_word_counts(text_matrix)

	# getting the number of words in each class
	total_words_class = np.sum(word_count,axis=1)
	print(total_words_class)
	# prints the count of each class
	# print(count_arr)


	return

if __name__ == "__main__":
 	main()